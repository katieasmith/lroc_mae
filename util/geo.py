"""
GeoTIFF helpers: pixel -> (lat, lon), acquisition date, and window reads for the main image
and for co-registered SPICE rasters.

Footprint (lat/lon) sources, tried in order:
  1. Corner tags in the main GeoTIFF (e.g. UPPER_LEFT_LATITUDE / UPPER_LEFT_LONGITUDE ...,
     as in the LROC PDS labels). Pixel lat/lon is bilinearly interpolated from the four corners.
  2. GCPs stored in the GeoTIFF (affine fitted to the GCPs, then projected -> geodetic).
  3. The GeoTIFF's geotransform + CRS (exact; works for lunar CRSs such as IAU_2015:30100
     or a custom Moon WKT, via pyproj's geodetic_crs).
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_gcps
from rasterio.windows import Window

# Images located by corner tags alone are expected to lack a geotransform.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

DEFAULT_CORNER_TAGS = {
    "UL": ("UPPER_LEFT_LATITUDE", "UPPER_LEFT_LONGITUDE"),
    "UR": ("UPPER_RIGHT_LATITUDE", "UPPER_RIGHT_LONGITUDE"),
    "LR": ("LOWER_RIGHT_LATITUDE", "LOWER_RIGHT_LONGITUDE"),
    "LL": ("LOWER_LEFT_LATITUDE", "LOWER_LEFT_LONGITUDE"),
}


def wrap_lon(lon):
    """Wrap longitude to [-180, 180)."""
    return (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _lookup_tag(tags: dict, key: str):
    """Case-insensitive tag lookup that tolerates a namespace prefix and units ("12.3 <deg>")."""
    lk = key.lower()
    for k, v in tags.items():
        if k.lower() == lk or k.lower().endswith(":" + lk):
            return v
    return None


def _to_float(v):
    if v is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))
    return float(m.group()) if m else None


class GeoLocator:
    """Maps full-image pixel coordinates (row, col; pixel centres at integer + 0.5 offsets
    are handled by the caller) to geodetic (lat, lon) in degrees."""

    def __init__(self, src: rasterio.io.DatasetReader, corner_tags: dict | None = None):
        self.H, self.W = src.height, src.width
        corner_tags = corner_tags or DEFAULT_CORNER_TAGS
        tags = src.tags()
        corners = {}
        for name, (lat_key, lon_key) in corner_tags.items():
            lat, lon = _to_float(_lookup_tag(tags, lat_key)), _to_float(_lookup_tag(tags, lon_key))
            if lat is not None and lon is not None:
                corners[name] = (lat, lon)

        if len(corners) == 4:
            self.mode = "corners"
            lat = np.array([corners[k][0] for k in ("UL", "UR", "LR", "LL")])
            lon = np.array([corners[k][1] for k in ("UL", "UR", "LR", "LL")])
            lon = lon[0] + wrap_lon(lon - lon[0])        # unwrap across the +-180 seam before interpolating
            self._corners = (lat, lon)
            return

        gcps, gcp_crs = src.gcps
        if gcps:
            transform, crs = from_gcps(gcps), gcp_crs
        elif src.crs is not None and not src.transform.is_identity:
            transform, crs = src.transform, src.crs
        else:
            raise ValueError(f"{src.name}: no corner tags, GCPs or georeferencing found")

        from pyproj import CRS, Transformer
        pcrs = CRS.from_wkt(crs.to_wkt())
        self.mode = "affine"
        self._transform = transform
        self._to_geo = Transformer.from_crs(pcrs, pcrs.geodetic_crs, always_xy=True)

    def latlon(self, rows, cols):
        """rows/cols: arrays of (fractional) pixel coordinates in the full image, where
        (0, 0) is the centre of the upper-left pixel. Returns (lat, lon) arrays in degrees."""
        rows, cols = np.asarray(rows, np.float64), np.asarray(cols, np.float64)
        if self.mode == "corners":
            u = cols / max(self.W - 1, 1)
            v = rows / max(self.H - 1, 1)
            (la_ul, la_ur, la_lr, la_ll), (lo_ul, lo_ur, lo_lr, lo_ll) = self._corners
            w_ul, w_ur, w_lr, w_ll = (1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v
            lat = w_ul * la_ul + w_ur * la_ur + w_lr * la_lr + w_ll * la_ll
            lon = w_ul * lo_ul + w_ur * lo_ur + w_lr * lo_lr + w_ll * lo_ll
            return lat, wrap_lon(lon)
        a = self._transform
        x = a.a * (cols + 0.5) + a.b * (rows + 0.5) + a.c
        y = a.d * (cols + 0.5) + a.e * (rows + 0.5) + a.f
        lon, lat = self._to_geo.transform(x, y)
        return np.asarray(lat), wrap_lon(lon)

    def footprint(self, n: int = 5):
        """(lat_min, lat_max, lon_min, lon_max, center_lat, center_lon) from an n x n grid.
        lon_min/lon_max are unwrapped relative to the centre, so lon_max may exceed 180."""
        r, c = np.meshgrid(np.linspace(0, self.H - 1, n), np.linspace(0, self.W - 1, n), indexing="ij")
        lat, lon = self.latlon(r.ravel(), c.ravel())
        clat, clon = self.latlon(np.array([(self.H - 1) / 2]), np.array([(self.W - 1) / 2]))
        lon = clon[0] + wrap_lon(lon - clon[0])
        return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()), float(clat[0]), float(clon[0])


# ----------------------------------------------------------------------------------------
# Dates
# ----------------------------------------------------------------------------------------
def parse_date(s) -> tuple[float, float] | None:
    """Return (year, day_of_year) from ISO ('2011-05-03T12:00:00'), TIFF ('2011:05:03 12:00:00')
    or PDS ordinal ('2011-123T12:00:00') strings."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().strip('"').rstrip("Z")
    m = re.match(r"^(\d{4})-(\d{3})(?:[T ].*)?$", s)                     # PDS ordinal date
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^(\d{4}):(\d{2}):(\d{2})(.*)$", s)                    # TIFFTAG_DATETIME
    if m:
        s = f"{m.group(1)}-{m.group(2)}-{m.group(3)}{m.group(4)}"
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)                                # >6 fractional digits
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return float(dt.year), float(dt.timetuple().tm_yday)


def date_from_tags(src, keys) -> tuple[float, float] | None:
    tags = src.tags()
    for k in keys or []:
        d = parse_date(_lookup_tag(tags, k))
        if d is not None:
            return d
    return None


# ----------------------------------------------------------------------------------------
# Window reads
# ----------------------------------------------------------------------------------------
def invalid_mask(arr: np.ndarray, nodata, valid_range=None) -> np.ndarray:
    """True where a pixel is unusable (nodata, non-finite or outside valid_range).
    arr: (bands, h, w). Returns (h, w); a pixel is invalid if any band is invalid."""
    bad = ~np.isfinite(arr) | (np.abs(arr) > 1e30)          # incl. ISIS special pixels
    if nodata is not None and np.isfinite(nodata):
        bad |= arr == nodata
    if valid_range is not None:
        lo, hi = valid_range
        bad |= (arr < lo) | (arr > hi)
    return bad.any(axis=0)


def read_image_window(src, row: int, col: int, size: int, bands, valid_range=None):
    """Returns (image float32 (C, size, size), invalid bool (size, size))."""
    arr = src.read(indexes=list(bands), window=Window(col, row, size, size)).astype(np.float32)
    return arr, invalid_mask(arr, src.nodata, valid_range)


# ----------------------------------------------------------------------------------------
# Auxiliary (SPICE / backplane) rasters
# ----------------------------------------------------------------------------------------
def _georeferenced(src) -> bool:
    return src.crs is not None and not src.transform.is_identity


def _clean(a: np.ndarray, nodata) -> np.ndarray:
    """nodata, non-finite and ISIS special pixels (|v| > 1e30) -> NaN."""
    a = a.astype(np.float32)
    bad = ~np.isfinite(a) | (np.abs(a) > 1e30)
    if nodata is not None and np.isfinite(nodata):
        bad |= a == nodata
    a[bad] = np.nan
    return a


class AuxGrid:
    """Maps main-image pixel positions to pixel indices of an auxiliary raster, whatever
    its grid or projection:

      same      identical grid (shape, transform, CRS) -> direct indexing
      georef    both georeferenced: main pixel -> main CRS -> (pyproj) aux CRS -> aux pixel.
                Covers backplanes projected with a different map (e.g. cam2map's default
                sinusoidal template) or at a coarser resolution or different extent.
      fraction  neither/one georeferenced: assumes both cover the same footprint

    Sampling is nearest-neighbour, so angles are never interpolated across the 0/360 wrap.
    """

    def __init__(self, main, aux):
        self.H, self.W = main.height, main.width
        self.aH, self.aW = aux.height, aux.width
        if (aux.height, aux.width) == (main.height, main.width) and aux.transform == main.transform \
                and aux.crs == main.crs:
            self.mode = "same"
        elif _georeferenced(main) and _georeferenced(aux):
            self.mode = "georef"
            self._fwd, self._inv, self._tf = main.transform, ~aux.transform, None
            if aux.crs != main.crs:
                from pyproj import CRS, Transformer
                self._tf = Transformer.from_crs(CRS.from_wkt(main.crs.to_wkt()), CRS.from_wkt(aux.crs.to_wkt()),
                                                always_xy=True)
        else:
            self.mode = "fraction"
        self.key = (self.mode, aux.crs.to_wkt() if aux.crs else None, tuple(aux.transform), aux.height, aux.width)

    def indices_at(self, rows, cols):
        """rows/cols: main-image pixel coordinates (0 = centre of the first pixel).
        Returns integer (row, col) indices into the aux raster; -1 where undefined."""
        rows, cols = np.asarray(rows, np.float64), np.asarray(cols, np.float64)
        if self.mode == "same":
            return np.rint(rows).astype(np.int64), np.rint(cols).astype(np.int64)
        if self.mode == "fraction":
            return (np.floor((rows + 0.5) * self.aH / self.H).astype(np.int64),
                    np.floor((cols + 0.5) * self.aW / self.W).astype(np.int64))
        a = self._fwd
        x = a.a * (cols + 0.5) + a.b * (rows + 0.5) + a.c
        y = a.d * (cols + 0.5) + a.e * (rows + 0.5) + a.f
        if self._tf is not None:
            x, y = (np.asarray(v) for v in self._tf.transform(x, y))
        b = self._inv
        c = b.a * x + b.b * y + b.c
        r = b.d * x + b.e * y + b.f
        ok = np.isfinite(r) & np.isfinite(c)
        r = np.where(ok, np.floor(np.where(ok, r, 0)), -1).astype(np.int64)
        c = np.where(ok, np.floor(np.where(ok, c, 0)), -1).astype(np.int64)
        return r, c


def sample_aux(src, ri: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """Nearest-neighbour values of band 1 at aux indices (ri, ci); NaN outside / nodata."""
    out = np.full(ri.shape, np.nan, np.float32)
    inside = (ri >= 0) & (ri < src.height) & (ci >= 0) & (ci < src.width)
    if not inside.any():
        return out
    r0, r1 = int(ri[inside].min()), int(ri[inside].max()) + 1
    c0, c1 = int(ci[inside].min()), int(ci[inside].max()) + 1
    a = _clean(src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)), src.nodata)
    out[inside] = a[ri[inside] - r0, ci[inside] - c0]
    return out


def tile_aux_indices(grid: AuxGrid, row: int, col: int, size: int, step: int = 4):
    """Aux indices for a size x size tile, evaluated every `step` pixels (block centres)."""
    k = np.arange(size // step) * step + (step - 1) / 2.0
    rr, cc = np.meshgrid(row + k, col + k, indexing="ij")
    return grid.indices_at(rr, cc)


def read_aux_tile(src, ri, ci, step: int = 4) -> np.ndarray:
    """(size, size) float32 array of aux values for a tile, NaN where unavailable."""
    a = sample_aux(src, ri, ci)
    return np.repeat(np.repeat(a, step, axis=0), step, axis=1)
