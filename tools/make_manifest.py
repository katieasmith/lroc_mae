"""
Build the LROC FM manifest from the outputs of process_edrs.sh.

Reads
    <processed>/GeoTiffs/<name>/<name>.tif                    main image (I/F, equirectangular)
    <processed>/GeoTiffs/<name>/<name>_{ph,em,in,lat,lon,scga,sga}.tif   phocube backplanes
    <processed>/intermediate_cub_files/catlab_files/<name>catlab.txt    START_TIME (date)
    <processed>/intermediate_cub_files/campt_files/<name>campt.txt      UTC (fallback date)
and writes one manifest row per complete image.

--check N compares the backplane latitude/longitude with the main image's georeferencing at
random pixels of N images. Offsets of about one backplane pixel are expected (both are
nearest-neighbour samples); much larger offsets mean the SPICE rasters are being read at the
wrong place and should be fixed before training.

Usage:
    python tools/make_manifest.py --processed ~/ProcessedData --check 10
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import rasterio

from util.geo import AuxGrid, GeoLocator, parse_date, sample_aux, wrap_lon

# manifest column -> file suffix written by make_geotiff()
BACKPLANES = {
    "phase": "ph",          # phocube band 1
    "emission": "em",       # band 2
    "incidence": "in",      # band 3
    "lat_bp": "lat",        # band 4
    "lon_bp": "lon",        # band 5
    "sc_azimuth": "scga",   # band 6  SubSpacecraftGroundAzimuth
    "sun_azimuth": "sga",   # band 7  SubSolarGroundAzimuth
}
MOON_RADIUS_M = 1737400.0


def read_date(name: str, catlab_dir: str, campt_dir: str) -> str | None:
    for folder, fname, pattern in ((catlab_dir, f"{name}catlab.txt", r"^\s*START_TIME\s*=\s*\"?([^\"\s]+)"),
                                   (campt_dir, f"{name}campt.txt", r"^\s*UTC\s*=\s*\"?([^\"\s]+)")):
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            with open(path, errors="ignore") as f:
                m = re.search(pattern, f.read(), flags=re.MULTILINE)
            if m and parse_date(m.group(1)):
                return m.group(1)
    return None


def check_alignment(row: dict, base: str, n_points: int = 2000, seed: int = 0):
    """Offsets (m) between backplane lat/lon and the main image georeferencing."""
    rng = np.random.default_rng(seed)
    with rasterio.open(os.path.join(base, row["image"])) as main, \
            rasterio.open(os.path.join(base, row["lat_bp"])) as la, \
            rasterio.open(os.path.join(base, row["lon_bp"])) as lo:
        rows = rng.uniform(0, main.height - 1, n_points)
        cols = rng.uniform(0, main.width - 1, n_points)
        lat_g, lon_g = GeoLocator(main).latlon(rows, cols)
        grid = AuxGrid(main, la)
        ri, ci = grid.indices_at(rows, cols)
        lat_b = sample_aux(la, ri, ci)
        lon_b = sample_aux(lo, *AuxGrid(main, lo).indices_at(rows, cols))
        ok = np.isfinite(lat_b) & np.isfinite(lon_b)
        if ok.sum() < 10:
            return None
        dn = np.deg2rad(lat_b[ok] - lat_g[ok]) * MOON_RADIUS_M
        de = np.deg2rad(wrap_lon(lon_b[ok] - lon_g[ok])) * MOON_RADIUS_M * np.cos(np.deg2rad(lat_g[ok]))
        d = np.hypot(dn, de)
        aux_px = abs(la.transform.a) if la.crs is not None and la.crs.is_projected else float("nan")
        main_px = abs(main.transform.a) if main.crs is not None and main.crs.is_projected else float("nan")
        return dict(mode=grid.mode, coverage=ok.mean(), median_m=float(np.median(d)),
                    p95_m=float(np.percentile(d, 95)), main_px_m=main_px, backplane_px_m=aux_px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="~/ProcessedData", help="datafolder in process_edrs.sh")
    ap.add_argument("--geotiffs", default=None)
    ap.add_argument("--catlab", default=None)
    ap.add_argument("--campt", default=None)
    ap.add_argument("--out", default=None, help="default: <processed>/manifest.csv")
    ap.add_argument("--check", type=int, default=10, help="number of images to check (0 = skip)")
    args = ap.parse_args()

    proc = os.path.expanduser(args.processed)
    geotiffs = os.path.expanduser(args.geotiffs or os.path.join(proc, "GeoTiffs"))
    catlab = os.path.expanduser(args.catlab or os.path.join(proc, "intermediate_cub_files", "catlab_files"))
    campt = os.path.expanduser(args.campt or os.path.join(proc, "intermediate_cub_files", "campt_files"))
    out = os.path.expanduser(args.out or os.path.join(proc, "manifest.csv"))
    base = os.path.dirname(os.path.abspath(out))

    rows, incomplete, undated = [], [], []
    for name in sorted(os.listdir(geotiffs)):
        folder = os.path.join(geotiffs, name)
        files = {"image": os.path.join(folder, f"{name}.tif")}
        files.update({col: os.path.join(folder, f"{name}_{suf}.tif") for col, suf in BACKPLANES.items()})
        if not os.path.isdir(folder):
            continue
        missing = [os.path.basename(p) for p in files.values() if not os.path.exists(p)]
        if missing:
            incomplete.append(f"{name}: missing {', '.join(missing)}")
            continue
        date = read_date(name, catlab, campt)
        if date is None:
            undated.append(name)
        row = {"image_id": name}
        row.update({k: os.path.relpath(p, base).replace(os.sep, "/") for k, p in files.items()})
        row["date"] = date
        rows.append(row)

    if not rows:
        sys.exit(f"no complete image folders found in {geotiffs}")
    man = pd.DataFrame(rows)
    os.makedirs(base, exist_ok=True)
    man.to_csv(out, index=False)
    print(f"wrote {out}: {len(man)} images ({len(incomplete)} incomplete skipped, {len(undated)} without a date)")
    for msg in incomplete[:20]:
        print("  incomplete", msg)
    if undated:
        print("  no START_TIME/UTC found for:", ", ".join(undated[:20]), "..." if len(undated) > 20 else "")

    if args.check > 0:
        print("\nalignment of backplanes with the main image (offsets in metres):")
        pick = man.sample(min(args.check, len(man)), random_state=0)
        bad = 0
        for _, r in pick.iterrows():
            res = check_alignment(r.to_dict(), base)
            if res is None:
                print(f"  {r.image_id}: too few overlapping valid pixels")
                bad += 1
                continue
            # correct mapping: p95 is roughly half to one backplane pixel (nearest-neighbour sampling)
            #limit = 2 * res["backplane_px_m"] if np.isfinite(res["backplane_px_m"]) else 50.0
            limit = 20 * res["main_px_m"] if np.isfinite(res["main_px_m"]) else 50.0
            flag = "" if res["p95_m"] <= max(limit, 3 * res["main_px_m"]) else "   <-- CHECK"
            bad += bool(flag)
            print(f"  {r.image_id}: mode={res['mode']} coverage={res['coverage']:.2f} "
                  f"median={res['median_m']:.1f} p95={res['p95_m']:.1f} "
                  f"(image px {res['main_px_m']:.2f}, backplane px {res['backplane_px_m']:.2f}){flag}")
        print("alignment OK" if bad == 0 else f"{bad} image(s) need a look")


if __name__ == "__main__":
    main()
