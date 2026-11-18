"""
Tests for models/lroc_vit.py (building encoders/MAEs and loading pretrained checkpoints).
Run from the repository root:  python tests/test_lroc_vit.py
"""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import lroc_vit as V  # noqa: E402
from models.lroc_mae import LROCMAE, LROCViT  # noqa: E402

VARS = [dict(name="lat", group="location"), dict(name="lon", group="location", periodic=True),
        dict(name="incidence", group="illumination"),
        dict(name="sun_azimuth", group="illumination", periodic=True, direction=True)]
SMALL = dict(img_size=64)


def _pretrained_mae(tmp):
    """Train-free stand-in for a pretrained MAE, saved like util/misc.save_model does."""
    torch.manual_seed(0)
    mae = V._create_lroc("lroc_vit_tiny", encoder_only=False, variables=VARS, **SMALL)
    with torch.no_grad():
        mae.encoder.meta.scale.copy_(torch.tensor([0.31, 0.47]))      # so a skipped load would show
    path = os.path.join(tmp, "checkpoint-best.pth")
    torch.save({"model": mae.state_dict(), "epoch": 3, "args": {"model": "lroc_mae_vit_tiny_patch16"},
                "variables": VARS}, path)
    return mae, path


def test_encoder_from_mae_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        mae, path = _pretrained_mae(tmp)
        enc = V._create_lroc("lroc_vit_tiny", pretrained=True, ckpt_path=path, **SMALL)
        assert isinstance(enc, LROCViT) and enc.variables == VARS, "variables should come from the checkpoint"
        sd_mae, sd_enc = mae.encoder.state_dict(), enc.state_dict()
        assert set(sd_mae) == set(sd_enc), "encoder keys differ from the MAE's encoder keys"
        for k in sd_mae:
            assert torch.equal(sd_mae[k], sd_enc[k]), f"{k} was not loaded"
        x, meta = torch.randn(2, 1, 64, 64), torch.rand(2, 4, 4, 4) * 90
        mae.eval(), enc.eval()
        feats = enc(x, meta)                                    # forward = selected forward_features
        ref = mae.encoder.forward_features(x, meta)
        assert len(feats) == len(enc.blocks) and all(torch.equal(a, b) for a, b in zip(feats, ref))


def test_out_indices_and_image_features():
    enc = V.lroc_vit_tiny(variables=VARS, out_indices=[1, 3], **SMALL).eval()
    feats = enc(torch.randn(2, 1, 1, 64, 64), torch.rand(2, 4, 4, 4) * 90)
    assert len(feats) == 2 and feats[0].shape == (2, 17, 192)
    maps = enc.prepare_features_for_image_model(feats)
    assert maps[0].shape == (2, 192, 4, 4), maps[0].shape
    tokens = feats[0][:, 1:, :]                                 # token k = patch (k // 4, k % 4)
    assert torch.equal(maps[0][:, :, 2, 3], tokens[:, 2 * 4 + 3, :])


def test_mae_from_checkpoint_other_input_size():
    with tempfile.TemporaryDirectory() as tmp:
        mae, path = _pretrained_mae(tmp)
        mae2 = V._create_lroc("lroc_vit_tiny", pretrained=True, ckpt_path=path, encoder_only=False, img_size=96)
        assert isinstance(mae2, LROCMAE) and mae2.encoder.pos_embed.shape[1] == 1 + 36
        for k, v in mae.state_dict().items():
            if "pos_embed" not in k:
                assert torch.equal(v, mae2.state_dict()[k]), f"{k} was not loaded"
        loss, pred, mask = mae2(torch.randn(2, 1, 1, 96, 96), meta=torch.rand(2, 4, 6, 6) * 90)
        assert pred.shape == (2, 36, 256) and torch.isfinite(loss)


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
