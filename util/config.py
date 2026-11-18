"""Config loading and metadata-variable parsing (no torch dependency)."""

from __future__ import annotations

import os

import pandas as pd
import yaml


# ----------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------
def _set_nested(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply 'a.b.c=value' overrides (values parsed as YAML),
    e.g. --opts model.mask_ratio=0.6 model.patch_size=8 data.crop_size=192."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, val = item.split("=", 1)
        _set_nested(cfg, key, yaml.safe_load(val))
    for section, key in (("data", "manifest"), ("data", "out_dir"), ("train", "out_dir")):
        if isinstance(cfg.get(section, {}).get(key), str):
            cfg[section][key] = os.path.expanduser(cfg[section][key])
    return cfg


def metadata_variables(cfg: dict) -> list[dict]:
    """Flatten metadata.groups into an ordered list of variable specs.

    Each variable: name, group, source in {geo, date, raster, column, tag}, and optional
    periodic (angle in degrees), direction (azimuth measured clockwise from image-up; reflected
    when the image is flipped), column (manifest column, defaults to name), key (tag name).
    """
    out = []
    for group, variables in (cfg.get("metadata", {}).get("groups") or {}).items():
        for v in variables:
            v = dict(v)
            v["group"] = group
            v.setdefault("periodic", False)
            v.setdefault("direction", False)
            v.setdefault("column", v["name"])
            if v["source"] == "geo":
                assert v["name"] in ("lat", "lon"), "geo variables must be named lat or lon"
            if v["source"] == "date":
                assert v["name"] in ("year", "doy"), "date variables must be named year or doy"
            out.append(v)
    return out


def metadata_groups(variables: list[dict]) -> list[str]:
    return list(dict.fromkeys(v["group"] for v in variables))


def read_manifest(path: str, variables: list[dict]) -> pd.DataFrame:
    """Read the manifest; add image_id from the file stem if absent; resolve relative paths
    (main image and raster-sourced variables) against the manifest's folder."""
    man = pd.read_csv(path, dtype={"image_id": str})
    if "image_id" not in man:
        man["image_id"] = [os.path.splitext(os.path.basename(p))[0] for p in man["image"]]
    base = os.path.dirname(os.path.abspath(path))
    cols = ["image"] + [v["column"] for v in variables if v["source"] == "raster" and v["column"] in man]
    for col in cols:
        man[col] = [p if os.path.isabs(str(p)) else os.path.normpath(os.path.join(base, str(p))) for p in man[col]]
    return man
