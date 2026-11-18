"""
Config handling and the PyTorch dataset that serves LROC tiles with per-patch metadata.

Each sample is a dict:
    image       (C, 1, S, S) float32   normalised image, nodata pixels set to 0
    valid       (1, 1, S, S) float32   1 = usable pixel (excluded from the loss otherwise)
    meta        (V, S/p, S/p) float32  one value per patch for each metadata variable
    meta_avail  (G,) bool              False if a metadata group is missing for this sample
where S is the crop size, p the patch size, V the number of variables, G the number of groups.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from .config import load_config, metadata_groups, metadata_variables, read_manifest  # noqa: F401
from .geo import AuxGrid, GeoLocator, tile_aux_indices, read_aux_tile, read_image_window


# ----------------------------------------------------------------------------------------
# Pooling helpers
# ----------------------------------------------------------------------------------------
def _pool(a: np.ndarray, p: int, periodic: bool) -> np.ndarray:
    """(S, S) -> (S/p, S/p) patch means (circular mean for angles), NaN-aware."""
    S = a.shape[0]
    blocks = a.reshape(S // p, p, S // p, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if periodic:
            r = np.deg2rad(blocks)
            return np.rad2deg(np.arctan2(np.nanmean(np.sin(r), axis=(1, 3)), np.nanmean(np.cos(r), axis=(1, 3))))
        return np.nanmean(blocks, axis=(1, 3))


def _fill_nan(g: np.ndarray, periodic: bool) -> bool:
    """Fill NaN patches in-place with the tile mean. Returns False if everything is NaN."""
    bad = ~np.isfinite(g)
    if not bad.any():
        return True
    if bad.all():
        g[:] = 0.0
        return False
    if periodic:
        r = np.deg2rad(g[~bad])
        g[bad] = np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    else:
        g[bad] = g[~bad].mean()
    return True


# ----------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------
class LROCTiles(Dataset):
    def __init__(self, cfg: dict, split: str = "train", train: bool = True, max_open_files: int = 64):
        d = cfg["data"]
        self.cfg = copy.deepcopy(cfg)
        self.train = train
        self.bands = list(d.get("bands", [1]))
        self.tile = int(d["tile_size"])
        self.crop = int(d["crop_size"])
        self.p = int(cfg["model"]["patch_size"])
        assert self.crop <= self.tile and self.crop % self.p == 0, "crop_size must be <= tile_size and divisible by patch_size"
        self.valid_range = d.get("valid_range")
        self.hflip = bool(d.get("hflip", True)) and train
        self.vflip = bool(d.get("vflip", False)) and train
        self.corner_tags = (cfg.get("geo") or {}).get("corner_tags")

        idx = d["out_dir"]
        tiles = pd.read_csv(os.path.join(idx, "tiles.csv"))
        self.tiles = tiles[tiles["split"] == split].reset_index(drop=True)
        self.images = pd.read_csv(os.path.join(idx, "images.csv"), dtype={"image_id": str}).set_index("image_id")
        self.variables = metadata_variables(cfg)
        self.groups = metadata_groups(self.variables)
        self.manifest = read_manifest(d["manifest"], self.variables).set_index("image_id")
        with open(os.path.join(idx, "stats.json")) as f:
            stats = json.load(f)
        self.mean = np.asarray(stats["mean"], np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], np.float32)[:, None, None]

        self._max_open = max_open_files
        self._handles: OrderedDict = OrderedDict()
        self._locators: dict = {}
        self._grids: OrderedDict = OrderedDict()
        self._pid = None
        self.aux_step = 4 if self.tile % 4 == 0 else 1

    def __len__(self):
        return len(self.tiles)

    # rasterio handles are opened lazily per worker process (not fork/spawn safe otherwise)
    def _open(self, path: str):
        if self._pid != os.getpid():
            self._handles.clear()
            self._locators.clear()
            self._grids.clear()
            self._pid = os.getpid()
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path]
        src = rasterio.open(path)
        self._handles[path] = src
        if len(self._handles) > self._max_open:
            _, old = self._handles.popitem(last=False)
            old.close()
        return src

    def _grid(self, main_path: str, main_src, aux_path: str, aux_src) -> AuxGrid:
        key = (main_path, aux_path)
        if key in self._grids:
            self._grids.move_to_end(key)
            return self._grids[key]
        grid = AuxGrid(main_src, aux_src)
        self._grids[key] = grid
        if len(self._grids) > 4 * self._max_open:
            self._grids.popitem(last=False)
        return grid

    def _locator(self, path: str, src) -> GeoLocator:
        if path not in self._locators:
            self._locators[path] = GeoLocator(src, self.corner_tags)
        return self._locators[path]

    def __getitem__(self, i: int):
        row = self.tiles.iloc[i]
        image_id, r0, c0 = str(row["image_id"]), int(row["row"]), int(row["col"])
        paths = self.manifest.loc[image_id]
        info = self.images.loc[image_id]
        src = self._open(paths["image"])
        x, bad = read_image_window(src, r0, c0, self.tile, self.bands, self.valid_range)
        x = (np.where(bad[None], self.mean, x) - self.mean) / self.std   # invalid -> 0 after normalising

        # augmentation parameters (torch RNG: correctly seeded per DataLoader worker)
        S, C = self.tile, self.crop
        if self.train:
            dy, dx = (int(v) for v in torch.randint(0, S - C + 1, (2,)))
        else:
            dy = dx = (S - C) // 2
        hf = self.hflip and bool(torch.rand(()) < 0.5)
        vf = self.vflip and bool(torch.rand(()) < 0.5)

        def aug(a):  # crop + flip on the last two axes
            a = a[..., dy:dy + C, dx:dx + C]
            if hf:
                a = a[..., :, ::-1]
            if vf:
                a = a[..., ::-1, :]
            return np.ascontiguousarray(a)

        x, bad = aug(x), aug(bad)

        # per-patch metadata
        hp = C // self.p
        meta = np.zeros((len(self.variables), hp, hp), np.float32)
        ok = {g: True for g in self.groups}
        latlon = None
        aux_idx = {}   # aux rasters on the same grid share one index computation
        for vi, v in enumerate(self.variables):
            src_type = v["source"]
            if src_type == "geo":
                if latlon is None:
                    centers = (np.arange(hp) + 0.5) * self.p - 0.5
                    rr, cc = np.meshgrid(r0 + dy + centers, c0 + dx + centers, indexing="ij")
                    lat, lon = self._locator(paths["image"], src).latlon(rr, cc)
                    latlon = {"lat": lat.astype(np.float32), "lon": lon.astype(np.float32)}
                    for k in latlon:   # lat/lon at patch centres of the crop; flip the grid like the image
                        if hf:
                            latlon[k] = latlon[k][:, ::-1]
                        if vf:
                            latlon[k] = latlon[k][::-1, :]
                g = np.array(latlon[v["name"]], np.float32)
            elif src_type == "raster":
                aux_path = paths[v["column"]]
                aux = self._open(aux_path)
                grid = self._grid(paths["image"], src, aux_path, aux)
                if grid.key not in aux_idx:
                    aux_idx[grid.key] = tile_aux_indices(grid, r0, c0, S, self.aux_step)
                a = read_aux_tile(aux, *aux_idx[grid.key], step=self.aux_step)
                g = _pool(aug(a), self.p, v["periodic"]).astype(np.float32)
            elif src_type == "date":
                g = np.full((hp, hp), info[v["name"]], np.float32)
            else:  # column / tag: image-level scalar resolved at index time
                g = np.full((hp, hp), info[f"meta_{v['name']}"], np.float32)

            if v["direction"]:   # azimuth clockwise from image-up: reflect with the image
                if hf:
                    g = (360.0 - g) % 360.0
                if vf:
                    g = (180.0 - g) % 360.0
            ok[v["group"]] &= _fill_nan(g, v["periodic"])
            meta[vi] = g

        return {
            "image": torch.from_numpy(x).unsqueeze(1),                         # (C, 1, S, S)
            "valid": torch.from_numpy(~bad).float()[None, None],               # (1, 1, S, S)
            "meta": torch.from_numpy(meta),                                    # (V, S/p, S/p)
            "meta_avail": torch.tensor([ok[g] for g in self.groups], dtype=torch.bool),
        }
