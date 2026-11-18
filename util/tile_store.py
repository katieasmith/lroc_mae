"""
Dataset over the arrays written by tools/extract_tiles.py. Returns exactly what the GeoTIFF dataset
returns (image, valid, meta, meta_avail), with the same random crop, flips, azimuth reflection,
normalisation and per-patch pooling, but reads only two memory-mapped arrays per sample.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import metadata_groups, metadata_variables

VAR_KEYS = ("name", "group", "source", "periodic", "direction")


def _weights(offset, crop, p, step, flip, G):
    """(crop/p, G) matrix: row i holds the fraction of patch i's pixels that fall in each stored grid cell.
    Averaging the stored grid with it equals averaging the grid repeated to pixel resolution, cropped and
    pooled over p x p patches (a flip reverses the patch order; pooling within a patch is order-free)."""
    rows = offset + np.arange(crop)
    w = np.zeros((crop // p, G), np.float64)
    np.add.at(w, (np.arange(crop) // p, rows // step), 1.0 / p)
    return w[::-1] if flip else w


def _pool(m, wr, wc, periodic):
    """m (V, G, G) with NaN = missing -> (V, h, w) NaN-aware weighted means; circular where periodic."""
    ok = np.isfinite(m)
    den = wr @ ok @ wc.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (wr @ np.where(ok, m, 0.0) @ wc.T) / den
        if periodic.any():
            r = np.deg2rad(np.where(ok[periodic], m[periodic], 0.0))
            s_ = wr @ np.sin(r) @ wc.T
            c_ = wr @ np.cos(r) @ wc.T
            out[periodic] = np.rad2deg(np.arctan2(s_, c_))
    out[den == 0] = np.nan
    return out


def _fill_nan(g, periodic):
    """Fill NaN patches with the tile mean (circular for angles). False if all NaN."""
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


class TileStore(Dataset):
    @staticmethod
    def path(cfg):
        """data.store in the config if set, else <data.out_dir>/store."""
        return os.path.expanduser(cfg["data"].get("store") or os.path.join(cfg["data"]["out_dir"], "store"))

    @classmethod
    def available(cls, cfg):
        return os.path.exists(os.path.join(cls.path(cfg), "store.json"))

    def __init__(self, cfg, split="train", train=True):
        d = cfg["data"]
        self.dir = self.path(cfg)
        with open(os.path.join(self.dir, "store.json")) as f:
            info = json.load(f)
        self.variables = metadata_variables(cfg)
        if [{k: v[k] for k in VAR_KEYS} for v in self.variables] != info["variables"]:
            raise ValueError(f"metadata variables in the config differ from {self.dir}/store.json: "
                             "rerun tools/extract_tiles.py --overwrite")
        self.groups = metadata_groups(self.variables)
        self.group_of = np.array([self.groups.index(v["group"]) for v in self.variables])
        self.periodic = np.array([bool(v["periodic"]) for v in self.variables])
        self.direction = np.array([bool(v["direction"]) for v in self.variables])

        self.split, self.train = split, train
        self.S, self.step = int(info["tile_size"]), int(info["step"])
        self.crop, self.p = int(d["crop_size"]), int(cfg["model"]["patch_size"])
        assert self.crop <= self.S and self.crop % self.p == 0, "crop_size must be <= tile size and divisible by patch_size"
        self.n = int(info["counts"][split])
        self.hflip = bool(d.get("hflip", True)) and train
        self.vflip = bool(d.get("vflip", False)) and train
        with open(os.path.join(d["out_dir"], "stats.json")) as f:
            stats = json.load(f)
        self.mean = np.asarray(stats["mean"], np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], np.float32)[:, None, None]
        self._img = self._meta = None
        self._pid = None

    def __len__(self):
        return self.n

    def _arrays(self):
        if self._pid != os.getpid():          # open the memory maps in each worker process
            self._img = np.load(os.path.join(self.dir, f"{self.split}_image.npy"), mmap_mode="r")
            self._meta = np.load(os.path.join(self.dir, f"{self.split}_meta.npy"), mmap_mode="r")
            self._pid = os.getpid()
        return self._img, self._meta

    def __getitem__(self, i):
        img, meta = self._arrays()
        x = np.asarray(img[i], dtype=np.float32)                    # (C, S, S), NaN = invalid
        m = np.asarray(meta[i], dtype=np.float32)                   # (V, S/k, S/k)
        bad = np.isnan(x).any(axis=0)
        x = (np.where(bad[None], self.mean, x) - self.mean) / self.std

        S, C = self.S, self.crop                                   # same draws, same order as the GeoTIFF dataset
        if self.train:
            dy, dx = (int(v) for v in torch.randint(0, S - C + 1, (2,)))
        else:
            dy = dx = (S - C) // 2
        hf = self.hflip and bool(torch.rand(()) < 0.5)
        vf = self.vflip and bool(torch.rand(()) < 0.5)

        def aug(a):
            a = a[..., dy:dy + C, dx:dx + C]
            if hf:
                a = a[..., :, ::-1]
            if vf:
                a = a[..., ::-1, :]
            return np.ascontiguousarray(a)

        x, bad = aug(x), aug(bad)
        G = S // self.step
        wr = _weights(dy, C, self.p, self.step, vf, G)
        wc = _weights(dx, C, self.p, self.step, hf, G)
        g = _pool(m.astype(np.float64), wr, wc, self.periodic).astype(np.float32)   # (V, C/p, C/p)

        if hf:
            g[self.direction] = (360.0 - g[self.direction]) % 360.0
        if vf:
            g[self.direction] = (180.0 - g[self.direction]) % 360.0
        avail = np.ones(len(self.groups), bool)
        for vi in range(len(self.variables)):
            if not _fill_nan(g[vi], self.periodic[vi]):
                avail[self.group_of[vi]] = False

        return {
            "image": torch.from_numpy(x).unsqueeze(1),                       # (C, 1, crop, crop)
            "valid": torch.from_numpy(~bad).float()[None, None],             # (1, 1, crop, crop)
            "meta": torch.from_numpy(g),                                     # (V, crop/p, crop/p)
            "meta_avail": torch.from_numpy(avail),
        }