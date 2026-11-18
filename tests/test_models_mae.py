"""
Tests for models_mae.py (step 2). Works for the 3D (Prithvi-style) and the 2D (MAE-style) version.

Reference files, in addition to step 1's tests/ref/mae_pos_embed.py and tests/ref/prithvi_mae.py:
    curl -sL https://raw.githubusercontent.com/facebookresearch/mae/main/models_mae.py -o tests/ref/mae_models_mae.py
    pip install einops        # only needed to import Prithvi's prithvi_mae.py

Run from the repository root. An optional argument runs only the tests whose name contains it:
    python tests/test_models_mae.py
    python tests/test_models_mae.py equivalence

Names the tests rely on (as in the walkthrough): LROCMAE(img_size, patch_size, in_chans, embed_dim, depth,
num_heads, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, norm_pix_loss, variables,
meta_mode); model.encoder.{patch_embed.proj, cls_token, blocks, norm, meta};
model.decoder.{decoder_embed, mask_token, decoder_blocks, decoder_norm, decoder_pred, meta};
meta.scale / meta.groups; model.patchify / unpatchify / forward_loss;
module-level random_masking(x, mask_ratio, noise=None, generator=None) as in Prithvi;
forward(imgs, valid=None, meta=None, meta_avail=None, mask_ratio=0.75, generator=None) -> (loss, pred, mask).
"""

import importlib
import importlib.util
import os
import sys
import types
from functools import partial

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "tests", "ref")
sys.path.insert(0, ROOT)
M = importlib.import_module(os.environ.get("LROC_MODEL_MODULE", "models_mae"))

SMALL = dict(img_size=64, patch_size=16, embed_dim=64, depth=2, num_heads=4,
             decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4, mlp_ratio=4.0)
VARS_TL = [dict(name="lat", group="location"), dict(name="lon", group="location"),
           dict(name="year", group="time"), dict(name="doy", group="time")]
VARS_LROC = [dict(name="lat", group="location"), dict(name="lon", group="location", periodic=True),
             dict(name="incidence", group="illumination"),
             dict(name="sun_azimuth", group="illumination", periodic=True, direction=True),
             dict(name="emission", group="viewing"), dict(name="phase", group="viewing"),
             dict(name="sc_azimuth", group="viewing", periodic=True, direction=True)]


def build(in_chans=1, **kw):
    return M.LROCMAE(in_chans=in_chans, **SMALL, **kw)


def is_3d(model):
    return isinstance(model.encoder.patch_embed.proj, nn.Conv3d)


def image(model, B=2, C=1, H=64, seed=0, smooth=False):
    g = torch.Generator().manual_seed(seed)
    shape = (B, C, 1, H, H) if is_3d(model) else (B, C, H, H)
    x = torch.randn(shape, generator=g)
    return x.cumsum(-1).cumsum(-2) / H if smooth else x


def close(a, b, atol, what=""):
    err = (a.detach().double() - b.detach().double()).abs().max().item()
    assert err <= atol, f"{what} max abs difference {err:.3g} > {atol}"


# ---- reference implementations ------------------------------------------------------------
def load_prithvi():
    path = os.path.join(REF, "prithvi_mae.py")
    if not os.path.exists(path):
        return None, "tests/ref/prithvi_mae.py missing (see step 1)"
    if importlib.util.find_spec("einops") is None:
        return None, "pip install einops to import Prithvi's model"
    spec = importlib.util.spec_from_file_location("prithvi_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, None


def load_mae():
    path, pos = os.path.join(REF, "mae_models_mae.py"), os.path.join(REF, "mae_pos_embed.py")
    if not (os.path.exists(path) and os.path.exists(pos)):
        return None, "tests/ref/mae_models_mae.py or mae_pos_embed.py missing"
    pos_mod = types.ModuleType("mae_pos_embed")
    exec(open(pos).read().replace("dtype=np.float)", "dtype=np.float64)"), pos_mod.__dict__)
    src = (open(path).read().replace("qk_scale=None, ", "")          # removed from timm's Block
           .replace("from util.pos_embed import get_2d_sincos_pos_embed", ""))
    mod = types.ModuleType("mae_ref")
    mod.get_2d_sincos_pos_embed = pos_mod.get_2d_sincos_pos_embed
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod, None


PRITHVI_GROUP = {"location": "location_embed", "time": "temporal_embed"}


def copy_from(ours, ref, strip_prefix=False):
    """Load the reference model's weights into ours, matching parameter names."""
    ref_sd, sd, unmatched = ref.state_dict(), ours.state_dict(), []
    for k, v in sd.items():
        name = k.split(".", 1)[1] if strip_prefix else k
        if name in ref_sd and ref_sd[name].shape == v.shape:
            sd[k] = ref_sd[name].clone()
        elif k.endswith("meta.scale"):           # our one-weight-per-group <- Prithvi's per-encoder scalars
            part, tag = ("encoder", "enc") if k.startswith("encoder") else ("decoder", "dec")
            groups = (ours.encoder if part == "encoder" else ours.decoder).meta.groups
            sd[k] = torch.tensor([ref_sd[f"{part}.{PRITHVI_GROUP[g]}_{tag}.scale"].item() for g in groups])
        else:
            unmatched.append(k)
    assert not unmatched, f"no counterpart in the reference for {unmatched[:5]}: rename to the MAE/Prithvi names"
    ours.load_state_dict(sd)


# ---- tests ----------------------------------------------------------------------------------
def test_factories_parameter_counts():
    expected = {"lroc_mae_vit_base_patch16": 85.25, "lroc_mae_vit_large_patch16": 302.58,
                "lroc_mae_vit_huge_patch14": 629.93}         # encoder, millions, in_chans=1
    for name, exp in expected.items():
        assert hasattr(M, name), f"missing factory {name}"
        with torch.device("meta"):
            m = getattr(M, name)(in_chans=1)
        n = sum(p.numel() for p in m.encoder.parameters()) / 1e6
        assert abs(n - exp) / exp < 0.01, f"{name}: {n:.1f}M encoder parameters, expected about {exp}M"


def test_patchify_layout():
    m = build(in_chans=2)
    x = image(m, C=2)
    p = m.patchify(x)
    assert p.shape == (2, 16, 16 * 16 * 2), p.shape
    assert torch.equal(m.unpatchify(p, x.shape[1:]), x), "unpatchify(patchify(x)) != x"
    img = x[:, :, 0] if is_3d(m) else x                      # (B, C, H, W)
    for k in (0, 1, 5, 15):                                  # token k = row k // 4, column k % 4, values in (p, q, c) order
        r, c = divmod(k, 4)
        block = img[:, :, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].permute(0, 2, 3, 1).reshape(2, -1)
        assert torch.equal(p[:, k], block), f"token {k} is not the patch at row {r}, column {c}"


def test_random_masking():
    assert hasattr(M, "random_masking"), "expected a module-level random_masking(x, mask_ratio, noise=None, generator=None)"
    x = torch.randn(3, 196, 8)
    xm, mask, ids_restore = M.random_masking(x, 0.75, generator=torch.Generator().manual_seed(0))
    assert xm.shape == (3, 49, 8) and (mask.sum(1) == 147).all(), "should keep 49 of 196 tokens"
    ids_shuffle = torch.argsort(ids_restore, dim=1)
    assert torch.equal(torch.gather(x, 1, ids_shuffle[:, :49, None].expand(-1, -1, 8)), xm), "kept tokens wrong"
    assert (mask.gather(1, ids_shuffle[:, :49]) == 0).all(), "mask must be 0 at kept tokens"
    _, mask2, _ = M.random_masking(x, 0.75, generator=torch.Generator().manual_seed(0))
    assert torch.equal(mask, mask2), "same generator seed must give the same mask"


def test_equivalence_with_reference():
    """Same weights + same random state => same mask, prediction and loss as MAE (2D) or Prithvi (3D)."""
    if is_3d(build()):
        P, why = load_prithvi()
        if P is None:
            return f"skipped ({why})"
        for npl in (False, True):
            torch.manual_seed(1)
            ref = P.PrithviMAE(patch_size=(1, 16, 16), num_frames=1, in_chans=1, norm_pix_loss=npl,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               **{k: v for k, v in SMALL.items() if k != "patch_size"}).eval()
            ours = build(norm_pix_loss=npl).eval()
            copy_from(ours, ref)
            x = image(ours)
            torch.manual_seed(123)
            lat, mask_r, ids = ref.encoder(x, None, None, 0.75)
            pred_r = ref.decoder(lat, ids, None, None, input_size=x.shape)
            loss_r = ref.forward_loss(x, pred_r, mask_r)
            torch.manual_seed(123)
            loss_o, pred_o, mask_o = ours(x, mask_ratio=0.75)
            assert torch.equal(mask_o, mask_r), "masks differ: random_masking must draw torch.rand(B, L) like Prithvi"
            close(pred_o, pred_r, 1e-5, f"prediction (norm_pix_loss={npl})")
            close(loss_o, loss_r, 1e-5, f"loss (norm_pix_loss={npl})")
        return "matches Prithvi-EO-2.0"
    A, why = load_mae()
    if A is None:
        return f"skipped ({why})"
    for npl in (False, True):
        torch.manual_seed(1)
        ref = A.MaskedAutoencoderViT(in_chans=3, norm_pix_loss=npl, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                     **SMALL).eval()
        ours = build(in_chans=3, norm_pix_loss=npl).eval()
        copy_from(ours, ref, strip_prefix=True)
        x = image(ours, C=3)                                   # MAE's patchify is hard-wired to 3 channels
        torch.manual_seed(123)
        loss_r, pred_r, mask_r = ref(x, mask_ratio=0.75)
        torch.manual_seed(123)
        loss_o, pred_o, mask_o = ours(x, mask_ratio=0.75)
        assert torch.equal(mask_o, mask_r), "masks differ: random_masking must draw torch.rand(B, L) like MAE"
        close(pred_o, pred_r, 1e-5, f"prediction (norm_pix_loss={npl})")
        close(loss_o, loss_r, 1e-5, f"loss (norm_pix_loss={npl})")
    return "matches MAE"


def test_equivalence_with_prithvi_metadata():
    """3D only: global mode with lat/lon/year/doy (non-periodic) must equal Prithvi's location/time encoders."""
    if not is_3d(build()):
        return "skipped (2D model; MAE has no metadata)"
    P, why = load_prithvi()
    if P is None:
        return f"skipped ({why})"
    torch.manual_seed(1)
    ref = P.PrithviMAE(patch_size=(1, 16, 16), num_frames=1, in_chans=1, coords_encoding=["time", "location"],
                       coords_scale_learn=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                       **{k: v for k, v in SMALL.items() if k != "patch_size"}).eval()
    with torch.no_grad():                                    # distinct weights, so a swapped mapping would show
        for name, val in (("encoder.location_embed_enc", .37), ("encoder.temporal_embed_enc", .61),
                          ("decoder.location_embed_dec", .23), ("decoder.temporal_embed_dec", .79)):
            ref.get_submodule(name).scale.fill_(val)
    ours = build(variables=VARS_TL, meta_mode="global").eval()
    copy_from(ours, ref)
    x = image(ours)
    vals = torch.tensor([[43.04, 210.03, 2014., 120.], [-80.5, 15.2, 2022., 300.]])
    meta = vals[:, :, None, None].expand(-1, -1, 4, 4).contiguous()
    loc, tim = vals[:, :2], vals[:, None, 2:]
    torch.manual_seed(7)
    lat, mask_r, ids = ref.encoder(x, tim, loc, 0.75)
    pred_r = ref.decoder(lat, ids, tim, loc, input_size=x.shape)
    loss_r = ref.forward_loss(x, pred_r, mask_r)
    torch.manual_seed(7)
    loss_o, pred_o, mask_o = ours(x, None, meta, mask_ratio=0.75)
    assert torch.equal(mask_o, mask_r)
    close(pred_o, pred_r, 1e-4, "prediction with metadata")
    close(loss_o, loss_r, 1e-5, "loss with metadata")
    return "matches Prithvi's location/time encoders"


def test_metadata_encoder():
    m = build(variables=VARS_LROC, meta_mode="patch").eval()
    enc = m.encoder.meta
    assert list(enc.groups) == ["location", "illumination", "viewing"], f"group order {enc.groups}"
    meta = torch.rand(2, 7, 4, 4) * 90
    e = enc(meta)
    assert e.shape == (2, 16, 64), e.shape
    m2 = meta.clone()
    m2[:, :, 0, 0] += 5                                      # patch mode: only token 0 changes
    d = (enc(m2) - e).abs().sum(-1)
    assert d[:, 0].min() > 0 and d[:, 1:].max() == 0, "changing one patch's metadata must change only that token"
    keep = torch.tensor([[True, False, True]] * 2)           # drop 'illumination'
    m3 = meta.clone()
    m3[:, 2:4] += 30
    close(enc(m3, keep), enc(meta, keep), 0, "dropped group still affects the output:")
    g = build(variables=VARS_LROC, meta_mode="global").encoder.meta(meta)
    close(g, g[:, :1].expand_as(g), 0, "global mode must give the same embedding to every token:")


def test_no_metadata_drop_and_gradients():
    """Metadata is always used, in training as in evaluation, and training adds no random draws."""
    x = image(build())
    meta = torch.rand(2, 7, 4, 4) * 90
    m = build(variables=VARS_LROC)
    assert not hasattr(m, "meta_drop"), "metadata drop should not exist"
    m.train()
    torch.manual_seed(3)
    l1, _, _ = m(x, None, meta)
    torch.manual_seed(3)
    l2, _, _ = m(x, None, meta + 50)
    assert (l1 - l2).abs() > 0, "in training the metadata must be used"
    m.eval()
    torch.manual_seed(3)
    l3, _, _ = m(x, None, meta)
    close(l1, l3, 1e-6, "training and evaluation should give the same loss (no extra randomness):")
    l3.backward()
    for part in (m.encoder, m.decoder):
        grad = part.meta.scale.grad
        assert grad is not None and grad.abs().sum() > 0, "metadata weights get no gradient"


def test_meta_avail_removes_groups():
    """A group marked unavailable has no effect; the others still do."""
    x, meta = image(build()), torch.rand(2, 7, 4, 4) * 90
    m = build(variables=VARS_LROC).train()
    avail = torch.tensor([[True, False, True]] * 2)           # location, illumination, viewing
    changed = meta.clone()
    changed[:, 2:4] += 30                                     # illumination variables only
    torch.manual_seed(4)
    a, _, _ = m(x, None, meta, avail)
    torch.manual_seed(4)
    b, _, _ = m(x, None, changed, avail)
    close(a, b, 0, "an unavailable group must not affect the loss:")


def test_loss_valid_mask():
    m = build()
    x = image(m)
    L, P = 16, 256
    pred = torch.randn(2, L, P)
    mask = torch.zeros(2, L)
    mask[:, ::2] = 1                                         # tokens 0, 2, ... masked
    target = m.patchify(x)
    mae = lambda mk: (((pred - target) ** 2).mean(-1) * mk).sum() / mk.sum()  # noqa: E731
    ones = torch.ones((2, 1) + tuple(x.shape[2:]))
    close(m.forward_loss(x, pred, mask, None), mae(mask), 1e-6, "valid=None must equal the MAE loss:")
    close(m.forward_loss(x, pred, mask, ones), mae(mask), 1e-6, "all-valid must equal the MAE loss:")
    v = ones.clone()
    v[0, ..., :16, :16] = 0                                  # token 0 of sample 0 fully invalid (and masked)
    v[1, ..., :8, 16:32] = 0                                 # half of token 1 of sample 1 (not masked)
    x2 = x.clone()
    x2[0, ..., :16, :16] = 1e6
    close(m.forward_loss(x, pred, mask, v), m.forward_loss(x2, pred, mask, v), 1e-6,
          "invalid pixels must not affect the loss:")
    mk = mask.clone()
    mk[0, 0] = 0
    close(m.forward_loss(x, pred, mask, v), mae(mk), 1e-6, "a fully invalid patch must be left out:")


def test_other_input_size():
    m = build().eval()
    loss, pred, mask = m(image(m, H=96))
    assert pred.shape[1] == 36 and torch.isfinite(loss), "96 x 96 input should give 36 tokens"


def test_time_axis_of_one_accepted():
    """The dataset returns (C, 1, H, W); a 2D model must give the same result as for (C, H, W)."""
    m = build(variables=VARS_LROC).eval()
    if is_3d(m):
        return "skipped (3D model)"
    x, meta = image(m), torch.rand(2, 7, 4, 4) * 90
    v = torch.ones(2, 1, 64, 64)
    torch.manual_seed(0)
    a = m(x, v, meta)
    torch.manual_seed(0)
    b = m(x.unsqueeze(2), v.unsqueeze(2), meta)
    close(a[0], b[0], 0, "loss:")
    close(a[1], b[1], 0, "prediction:")


def test_overfit_one_batch():
    torch.manual_seed(0)
    m = build(variables=VARS_LROC)
    x, meta = image(m, B=4, smooth=True), torch.rand(4, 7, 4, 4) * 90
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g, first = torch.Generator(), None
    for _ in range(150):
        g.manual_seed(0)                                     # same mask every step
        loss, _, _ = m(x, None, meta, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.25 * first, f"loss only went {first:.3f} -> {loss.item():.3f}"
    return f"{first:.3f} -> {loss.item():.4f}"


def test_bf16_on_gpu():
    if not torch.cuda.is_available():
        return "skipped (no GPU)"
    m = build(variables=VARS_LROC).cuda().eval()
    x, meta = image(m).cuda(), (torch.rand(2, 7, 4, 4) * 90).cuda()
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    l32, _, _ = m(x, None, meta, generator=g)
    g.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l16, _, _ = m(x, None, meta, generator=g)
    l16.backward()
    assert torch.isfinite(l16) and abs(l16.item() - l32.item()) / l32.item() < 0.05, (l32.item(), l16.item())
    return f"fp32 {l32.item():.4f}, bf16 {l16.item():.4f}"


if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        if pick not in name:
            continue
        try:
            note = fn()
            print(f"PASS  {name}" + (f"  {note}" if isinstance(note, str) else ""))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
