"""
Build the tile index for LROC FM pretraining.

Input: a manifest CSV with one row per image, e.g.

    image_id,image,incidence,emission,phase,sun_azimuth,date
    M104318871LE,D:/lroc/M104318871LE.tif,D:/lroc/M104318871LE_inc.tif,D:/lroc/M104318871LE_emi.tif,D:/lroc/M104318871LE_pha.tif,D:/lroc/M104318871LE_saz.tif,2009-08-27T05:51:20

  image      main GeoTIFF (image + footprint as corner tags, GCPs or geotransform/CRS)
  <name>     one column per metadata variable with source "raster" (path to the SPICE GeoTIFF)
             or source "column" (a number)
  date       optional; if absent, the tags listed in geo.date_tags are searched
  image_id   optional; defaults to the image file stem

Output (data.out_dir):
  images.csv   per-image footprint, date, scalar metadata and split
  tiles.csv    image_id, row, col, split, center_lat, center_lon
  stats.json   per-band mean/std over valid training pixels

Split (following Prithvi-EO-2.0, Sec. III-A): 95%/5% train/validation by image, assigned through
a lat/lon grid of `split_cell_deg` cells so validation images are spatially clustered; training
tiles that fall in any cell touched by a validation image are discarded, so repeat coverage of
the same terrain cannot leak into validation.

Usage:
    python tools/build_index.py --config configs/data.yaml --workers 8
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from util.config import load_config, metadata_variables, read_manifest
from util.geo import GeoLocator, _lookup_tag, _to_float, date_from_tags, invalid_mask, parse_date, wrap_lon


# ----------------------------------------------------------------------------------------
# Spatial cells
# ----------------------------------------------------------------------------------------
def cell_of(lat, lon, deg):
    lat = np.clip(np.asarray(lat, np.float64), -90, 89.999999)
    lon = wrap_lon(lon)
    return list(zip(np.floor((lat + 90) / deg).astype(int).tolist(), np.floor((lon + 180) / deg).astype(int).tolist()))


def cells_in_box(lat_min, lat_max, lon_min, lon_max, deg):
    lats = np.append(np.arange(lat_min, lat_max, deg / 2), lat_max)
    lons = np.append(np.arange(lon_min, lon_max, deg / 2), lon_max)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    return set(cell_of(la.ravel(), lo.ravel(), deg))


# ----------------------------------------------------------------------------------------
# Pass 1: per-image footprint and image-level metadata
# ----------------------------------------------------------------------------------------
def _scan_image(task):
    row, cfg, variables = task
    try:
        with rasterio.open(row["image"]) as src:
            loc = GeoLocator(src, (cfg.get("geo") or {}).get("corner_tags"))
            lat_min, lat_max, lon_min, lon_max, clat, clon = loc.footprint()
            date = parse_date(row.get("date")) or date_from_tags(src, (cfg.get("geo") or {}).get("date_tags"))
            out = dict(image_id=row["image_id"], height=src.height, width=src.width, count=src.count,
                       center_lat=clat, center_lon=clon, lat_min=lat_min, lat_max=lat_max,
                       lon_min=lon_min, lon_max=lon_max,
                       year=date[0] if date else np.nan, doy=date[1] if date else np.nan)
            tags = src.tags()
            for v in variables:
                if v["source"] == "column":
                    out[f"meta_{v['name']}"] = _to_float(row.get(v["column"]))
                elif v["source"] == "tag":
                    out[f"meta_{v['name']}"] = _to_float(_lookup_tag(tags, v.get("key", v["name"])))
            return out, None
    except Exception as e:  # keep going; report at the end
        return None, f"{row['image_id']}: {e}"


# ----------------------------------------------------------------------------------------
# Pass 2: tiling, filtering and statistics
# ----------------------------------------------------------------------------------------
def _tile_image(task):
    image_id, path, split, cfg, blocked_cells, seed = task
    d = cfg["data"]
    S, bands = int(d["tile_size"]), list(d.get("bands", [1]))
    max_nodata, min_std = float(d.get("max_nodata_frac", 0.01)), float(d.get("min_tile_std", 0.0))
    deg = float(d.get("split_cell_deg", 2.0))
    s1 = np.zeros(len(bands)); s2 = np.zeros(len(bands)); n = 0
    tiles, n_blocked = [], 0
    with rasterio.open(path) as src:
        loc = GeoLocator(src, (cfg.get("geo") or {}).get("corner_tags"))
        H, W = src.height, src.width
        ncol = W // S
        for r in range(0, H - S + 1, S) if ncol else []:
            strip = src.read(indexes=bands, window=Window(0, r, ncol * S, S)).astype(np.float32)
            for c in range(0, ncol * S, S):
                sub = strip[:, :, c:c + S]
                bad = invalid_mask(sub, src.nodata, d.get("valid_range"))
                if bad.mean() > max_nodata:
                    continue
                good = sub[:, ~bad]
                if min_std > 0 and good.std(axis=1).max() < min_std:
                    continue
                rr = np.array([r, r, r + S - 1, r + S - 1, r + (S - 1) / 2])
                cc = np.array([c, c + S - 1, c, c + S - 1, c + (S - 1) / 2])
                lat, lon = loc.latlon(rr, cc)
                if split == "train" and blocked_cells and set(cell_of(lat[:4], lon[:4], deg)) & blocked_cells:
                    n_blocked += 1
                    continue
                g64 = good.astype(np.float64)
                tiles.append((r, c, float(lat[4]), float(lon[4]), g64.sum(axis=1), (g64 ** 2).sum(axis=1), g64.shape[1]))
    rng = np.random.default_rng(seed)
    max_tiles = int(d.get("max_tiles_per_image", 1500))
    if len(tiles) > max_tiles:   # Prithvi: cap samples per tile, randomly downsample otherwise
        tiles = [tiles[i] for i in sorted(rng.choice(len(tiles), max_tiles, replace=False))]
    rows = []
    for r, c, la, lo, cs1, cs2, cn in tiles:
        if split == "train":
            s1 += cs1; s2 += cs2; n += cn
        rows.append(dict(image_id=image_id, row=r, col=c, split=split, center_lat=la, center_lon=lo))
    return rows, s1, s2, n, n_blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    d = cfg["data"]
    variables = metadata_variables(cfg)
    os.makedirs(d["out_dir"], exist_ok=True)

    man = read_manifest(d["manifest"], variables)
    missing = [v["column"] for v in variables if v["source"] in ("raster", "column") and v["column"] not in man]
    if missing:
        sys.exit(f"manifest is missing columns for metadata variables: {missing}")

    Pool = ProcessPoolExecutor if args.workers > 1 else None

    def run(fn, tasks):
        if Pool is None:
            return list(map(fn, tasks))
        with Pool(max_workers=args.workers) as ex:
            return list(ex.map(fn, tasks, chunksize=1))

    # ---- pass 1
    results = run(_scan_image, [(r, cfg, variables) for r in man.to_dict("records")])
    errors = [e for _, e in results if e]
    images = pd.DataFrame([o for o, _ in results if o])
    for e in errors:
        print("skipped", e)
    if images.empty:
        sys.exit("no readable images in the manifest")
    nb = len(d.get("bands", [1]))
    images = images[images["count"] >= max(d.get("bands", [1]))]
    print(f"{len(images)} images scanned ({len(errors)} skipped), {nb} band(s) used")

    # ---- spatial 95/5 split
    deg, rng = float(d.get("split_cell_deg", 2.0)), np.random.default_rng(d.get("seed", 0))
    images["cell"] = cell_of(images["center_lat"], images["center_lon"], deg)
    cells = images["cell"].unique().tolist()
    rng.shuffle(cells)
    target, val_cells, n_val = float(d.get("val_frac", 0.05)) * len(images), set(), 0
    counts = images["cell"].value_counts()
    for c in cells:
        if n_val >= target:
            break
        val_cells.add(c); n_val += counts[c]
    images["split"] = np.where(images["cell"].isin(val_cells), "val", "train")
    blocked = set()
    for _, im in images[images["split"] == "val"].iterrows():
        blocked |= cells_in_box(im["lat_min"], im["lat_max"], im["lon_min"], im["lon_max"], deg)
    print(f"split: {int((images.split == 'train').sum())} train / {int((images.split == 'val').sum())} val images, "
          f"{len(blocked)} blocked cells of {deg} deg")

    # ---- pass 2
    paths = man.set_index("image_id")["image"]
    tasks = [(im.image_id, paths[im.image_id], im.split, cfg, blocked, int(d.get("seed", 0)) * 100003 + k)
             for k, im in enumerate(images.itertuples())]
    out = run(_tile_image, tasks)
    tiles = pd.DataFrame([row for rows, *_ in out for row in rows])
    s1 = sum(o[1] for o in out); s2 = sum(o[2] for o in out); n = sum(o[3] for o in out)
    n_blocked = sum(o[4] for o in out)
    if tiles.empty or n == 0:
        sys.exit("no usable training tiles; check tile_size, max_nodata_frac, valid_range and the split")
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    images["n_tiles"] = images["image_id"].map(tiles.groupby("image_id").size()).fillna(0).astype(int)

    images.drop(columns=["cell"]).to_csv(os.path.join(d["out_dir"], "images.csv"), index=False)
    tiles.to_csv(os.path.join(d["out_dir"], "tiles.csv"), index=False)
    with open(os.path.join(d["out_dir"], "stats.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "n_pixels": int(n),
                   "bands": d.get("bands", [1]), "tile_size": d["tile_size"]}, f, indent=2)
    vc = tiles["split"].value_counts()
    print(f"tiles: {vc.get('train', 0)} train / {vc.get('val', 0)} val "
          f"({n_blocked} training tiles dropped for overlapping validation cells)")
    print(f"band mean {np.round(mean, 5).tolist()}  std {np.round(std, 5).tolist()}")


if __name__ == "__main__":
    main()
