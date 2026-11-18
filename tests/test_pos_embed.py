"""
Tests for util/pos_embed.py.

Reference implementations (downloaded once into tests/ref/):
    curl -sL https://raw.githubusercontent.com/facebookresearch/mae/main/util/pos_embed.py -o tests/ref/mae_pos_embed.py
    curl -sL https://raw.githubusercontent.com/IBM/terratorch/main/terratorch/models/backbones/prithvi_mae.py -o tests/ref/prithvi_mae.py

Run from the repository root:
    python tests/test_pos_embed.py          (or: python -m pytest tests -q)
"""

import ast
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from util import pos_embed as pe  # noqa: E402

REF = os.path.join(ROOT, "tests", "ref")


def load_ref(filename, names):
    """Take selected top-level functions from a reference file without importing it
    (prithvi_mae.py imports einops/timm; MAE uses np.float, removed in NumPy 1.24)."""
    path = os.path.join(REF, filename)
    if not os.path.exists(path):
        return None
    src = open(path).read().replace("dtype=np.float)", "dtype=np.float64)")
    ns = {"np": np, "torch": torch}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
    return ns


MAE = load_ref("mae_pos_embed.py", {"get_1d_sincos_pos_embed_from_grid", "get_2d_sincos_pos_embed",
                                    "get_2d_sincos_pos_embed_from_grid"})
PRI = load_ref("prithvi_mae.py", {"get_1d_sincos_pos_embed_from_grid", "get_3d_sincos_pos_embed",
                                  "_get_1d_sincos_embed_from_grid_torch"})


def close(a, b, atol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    err = np.abs(a - b).max()
    assert err <= atol, f"max abs difference {err:.3g} > {atol}"


# ---------------------------------------------------------------------------------------
# 1D (MAE)
# ---------------------------------------------------------------------------------------
def test_1d_matches_mae():
    assert MAE, "tests/ref/mae_pos_embed.py missing"
    for D in (64, 768):
        for pos in (np.arange(14), np.random.default_rng(0).uniform(-500, 2500, 50)):
            close(pe.get_1d_sincos_pos_embed_from_grid(D, pos),
                  MAE["get_1d_sincos_pos_embed_from_grid"](D, pos.astype(np.float64)), 1e-12)


def test_1d_properties():
    D, pos = 64, np.arange(20)
    e = pe.get_1d_sincos_pos_embed_from_grid(D, pos)
    assert e.shape == (20, D) and np.abs(e).max() <= 1
    close(e[:, :D // 2] ** 2 + e[:, D // 2:] ** 2, np.ones((20, D // 2)), 1e-12)   # (sin, cos) pairs
    # the dot product depends only on the offset: e(3).e(5) == e(10).e(12)
    close(e[3] @ e[5], e[10] @ e[12], 1e-9)
    assert len({tuple(np.round(r, 9)) for r in e}) == len(pos), "two positions share an encoding"


# ---------------------------------------------------------------------------------------
# 2D (MAE) -- only if you kept it
# ---------------------------------------------------------------------------------------
def test_2d_matches_mae():
    if not hasattr(pe, "get_2d_sincos_pos_embed"):
        return "skipped (no 2D function)"
    assert MAE, "tests/ref/mae_pos_embed.py missing"
    for D, g in ((768, 14), (192, 7)):
        close(pe.get_2d_sincos_pos_embed(D, g, cls_token=True), MAE["get_2d_sincos_pos_embed"](D, g, cls_token=True), 1e-12)


def test_2d_token_order():
    """Token k must be the patch at row k // W, column k % W (the order patchify uses).
    MAE's first half encodes the column (its 'emb_h' is built from the w grid), the second the row."""
    if not hasattr(pe, "get_2d_sincos_pos_embed"):
        return "skipped (no 2D function)"
    D, g = 64, 6
    e = pe.get_2d_sincos_pos_embed(D, g)
    one = pe.get_1d_sincos_pos_embed_from_grid
    k = np.arange(g * g)
    close(e[:, :D // 2], one(D // 2, k % g), 1e-12)
    close(e[:, D // 2:], one(D // 2, k // g), 1e-12)


def test_2d_rectangular_grid():
    """Only needed if your 2D function accepts (h, w); MAE's original is square-only."""
    if not hasattr(pe, "get_2d_sincos_pos_embed"):
        return "skipped (no 2D function)"
    D, (h, w) = 64, (3, 5)
    try:
        e = pe.get_2d_sincos_pos_embed(D, (h, w), cls_token=True)
    except TypeError:
        return "skipped (square grids only)"
    assert e.shape == (1 + h * w, D) and not e[0].any()
    one, k = pe.get_1d_sincos_pos_embed_from_grid, np.arange(h * w)
    close(e[1:, :D // 2], one(D // 2, k % w), 1e-12)
    close(e[1:, D // 2:], one(D // 2, k // w), 1e-12)


# ---------------------------------------------------------------------------------------
# 3D (Prithvi) -- only if you kept it
# ---------------------------------------------------------------------------------------
def test_3d_matches_prithvi():
    if not hasattr(pe, "get_3d_sincos_pos_embed"):
        return "skipped (no 3D function)"
    assert PRI, "tests/ref/prithvi_mae.py missing"
    for D, g in ((768, (1, 14, 14)), (1024, (1, 14, 14)), (192, (3, 7, 5))):
        close(pe.get_3d_sincos_pos_embed(D, g, cls_token=True),
              PRI["get_3d_sincos_pos_embed"](D, g, add_cls_token=True), 1e-12)


def test_3d_token_order():
    """Token k = (t, h, w) with w fastest; columns [w 6/16 | h 6/16 | t 4/16]."""
    if not hasattr(pe, "get_3d_sincos_pos_embed"):
        return "skipped (no 3D function)"
    D, (T, H, W) = 192, (2, 4, 5)
    e = pe.get_3d_sincos_pos_embed(D, (T, H, W), cls_token=True)
    assert e.shape == (1 + T * H * W, D) and not e[0].any(), "row 0 must be the all-zero cls row"
    e = e[1:]
    one = pe.get_1d_sincos_pos_embed_from_grid
    k = np.arange(T * H * W)
    wd, hd = D // 16 * 6, D // 16 * 6
    close(e[:, :wd], one(wd, k % W), 1e-12)
    close(e[:, wd:wd + hd], one(hd, (k // W) % H), 1e-12)
    close(e[:, wd + hd:], one(D - wd - hd, k // (H * W)), 1e-12)
    single = pe.get_3d_sincos_pos_embed(D, (1, H, W))[:, wd + hd:]
    assert np.allclose(single, single[0]), "with T = 1 the time block should be constant"


# ---------------------------------------------------------------------------------------
# Metadata encodings
# ---------------------------------------------------------------------------------------
def test_sincos_1d_matches_prithvi_and_mae():
    vals = torch.tensor([0.0, 43.04, -89.9, 210.03, 359.9, 2009.0, 2026.0, 365.0])
    for D in (128, 384):
        ours = pe.sincos_1d(vals, D)
        assert ours.shape == (len(vals), D) and ours.dtype == torch.float32
        if PRI:
            close(ours, PRI["_get_1d_sincos_embed_from_grid_torch"](D, vals.clone()), 1e-5)
        close(ours, pe.get_1d_sincos_pos_embed_from_grid(D, vals.double().numpy()), 1e-3)  # float32 vs float64
    assert pe.sincos_1d(vals.double(), 64).dtype == torch.float32, "should compute in float32"


def test_sincos_periodic():
    D = 384
    f = lambda a: pe.sincos_periodic(torch.tensor(a, dtype=torch.float32), D)  # noqa: E731
    a = [0.0, 37.5, 143.92, 248.5, 315.72, 359.9]
    assert f(a).shape == (len(a), D)
    close(f(a), f([x + 360 for x in a]), 2e-3)                 # exactly 360-degree periodic
    close(f(a), f([x - 360 for x in a]), 2e-3)
    close(f([359.9]), f([-0.1]), 2e-3)                         # no seam
    d = lambda x, y: (f([x]) - f([y])).norm().item()           # noqa: E731
    assert d(10, 10.5) < d(10, 190), "nearby angles should be closer than opposite ones"
    seam = (pe.sincos_1d(torch.tensor([359.9]), D) - pe.sincos_1d(torch.tensor([-0.1]), D)).norm().item()
    print(f"    (for comparison, sincos_1d puts 359.9 and -0.1 at distance {seam:.2f}; periodic: {d(359.9, -0.1):.4f})")


def test_odd_dim_rejected():
    for fn in (pe.sincos_1d, pe.sincos_periodic):
        try:
            out = fn(torch.tensor([1.0]), 63)
        except AssertionError:
            continue
        raise AssertionError(f"{fn.__name__}(dim=63) returned shape {tuple(out.shape)}; "
                             "add `assert dim % 2 == 0` so an odd split fails loudly")


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        try:
            note = fn()
            print(f"PASS  {name}" + (f"  {note}" if isinstance(note, str) else ""))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
