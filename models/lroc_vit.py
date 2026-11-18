"""
Build LROC FM encoders (for fine-tuning) and MAEs (for pretraining), and load pretrained checkpoints.

Follows terratorch/models/backbones/prithvi_vit.py: one config per variant, checkpoint filters that turn
an MAE checkpoint into encoder-only (or MAE) weights, a builder, and one factory function per variant.
Prithvi's band handling (select_patch_embed_weights) and Hugging Face download are not needed here.
"""

from __future__ import annotations

import logging

import warnings

import torch

from models.lroc_mae import LROCMAE, LROCViT

logger = logging.getLogger(__name__)

# TerraTorch is only needed to use these models through TerraTorch (fine-tuning). Pretraining and the tests run
# without it, so a missing or broken install (e.g. TerraTorch with a torchgeo that is too old) only skips registration.
try:
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_FULL_MODEL_REGISTRY

    register_backbone = TERRATORCH_BACKBONE_REGISTRY.register
    register_full_model = TERRATORCH_FULL_MODEL_REGISTRY.register
except Exception as e:  # noqa: BLE001  (a broken install raises AttributeError, not ImportError)
    warnings.warn(f"TerraTorch unavailable ({type(e).__name__}: {e}); LROC models are not registered with it.",
                  stacklevel=2)
    register_backbone = register_full_model = lambda fn: fn

def _cfg(**kwargs):
    return {"img_size": 224, "patch_size": 16, "in_chans": 1, "embed_dim": 768,
            "depth": 12, "num_heads": 12, "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16,
            "mlp_ratio": 4.0, "mask_ratio": 0.75, "norm_pix_loss": False, **kwargs}


lroc_cfgs = {
    "lroc_vit_tiny": _cfg(embed_dim=192, depth=4, num_heads=3, decoder_embed_dim=128, decoder_depth=2,
                          decoder_num_heads=4),                       # debugging
    "lroc_vit_base": _cfg(),                                          # MAE ViT-B, proof of concept
    "lroc_vit_large": _cfg(embed_dim=1024, depth=24, num_heads=16),   # Prithvi-EO-2.0-300M
    "lroc_vit_huge": _cfg(embed_dim=1280, depth=32, num_heads=16, patch_size=14),  # Prithvi-EO-2.0-600M
}


def load_checkpoint(path):
    """Accepts a training checkpoint ({'model': state_dict, 'variables': [...], ...}) or a bare state dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("variables")
    return ckpt, None


def checkpoint_filter_fn_vit(state_dict, model: LROCViT) -> dict:
    """MAE checkpoint -> encoder-only weights."""
    clean = {}
    for k, v in state_dict.items():
        if k.startswith("decoder.") or "mask_token" in k:
            continue                                     # drop decoder weights
        if k.startswith("encoder."):
            k = k[len("encoder."):]
        if k == "pos_embed":
            v = model.pos_embed                          # fixed; depends on the input size
        clean[k] = v
    return clean


def checkpoint_filter_fn_mae(state_dict, model: LROCMAE) -> dict:
    """MAE checkpoint -> MAE weights (positional embeddings taken from the new model)."""
    clean = dict(state_dict)
    if "encoder.pos_embed" in clean:
        clean["encoder.pos_embed"] = model.encoder.pos_embed
    if "decoder.decoder_pos_embed" in clean:
        clean["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    return clean


def _create_lroc(variant, pretrained=False, ckpt_path=None, encoder_only=True, variables=None, **kwargs):
    """Build an LROCViT (encoder_only=True) or LROCMAE. With pretrained=True, load ckpt_path.
    The metadata variables are taken from the checkpoint unless given."""
    model_args = lroc_cfgs[variant].copy()
    state_dict = None
    if pretrained:
        assert ckpt_path is not None, "pretrained=True needs ckpt_path"
        state_dict, ckpt_variables = load_checkpoint(ckpt_path)
        variables = ckpt_variables if variables is None else variables
    out_indices = kwargs.pop("out_indices", None)
    assert encoder_only or out_indices is None, "out_indices provided for an MAE model"
    model_args.update(kwargs)
    model_args["variables"] = variables
    model = (LROCViT if encoder_only else LROCMAE)(**model_args)

    if state_dict is not None:
        filt = checkpoint_filter_fn_vit if encoder_only else checkpoint_filter_fn_mae
        keys = model.load_state_dict(filt(state_dict, model), strict=False)
        if keys.missing_keys:
            logger.warning(f"Missing keys in {ckpt_path}: {keys.missing_keys}")
        if keys.unexpected_keys:
            logger.warning(f"Unexpected keys in {ckpt_path}: {keys.unexpected_keys}")
    elif ckpt_path is not None:
        logger.warning(f"ckpt_path is provided but pretrained is False; ignoring {ckpt_path}")

    if encoder_only:                                  # as in prithvi_vit.py: forward returns selected blocks
        out_indices = list(range(len(model.blocks))) if out_indices is None else list(out_indices)

        def forward_filter_indices(*args, **kw):
            features = model.forward_features(*args, **kw)
            return [features[i] for i in out_indices]

        model.forward = forward_filter_indices
        model.out_indices = out_indices
        model.out_channels = [model.out_channels[i] for i in out_indices]   # one entry per returned feature
    return model


@register_backbone
def lroc_vit_tiny(pretrained=False, **kwargs) -> LROCViT:
    return _create_lroc("lroc_vit_tiny", pretrained, **kwargs)


@register_backbone
def lroc_vit_base(pretrained=False, **kwargs) -> LROCViT:
    return _create_lroc("lroc_vit_base", pretrained, **kwargs)


@register_backbone
def lroc_vit_large(pretrained=False, **kwargs) -> LROCViT:
    return _create_lroc("lroc_vit_large", pretrained, **kwargs)


@register_backbone
def lroc_vit_huge(pretrained=False, **kwargs) -> LROCViT:
    return _create_lroc("lroc_vit_huge", pretrained, **kwargs)


@register_full_model
def lroc_vit_base_mae(pretrained=False, **kwargs) -> LROCMAE:
    return _create_lroc("lroc_vit_base", pretrained, encoder_only=False, **kwargs)


@register_full_model
def lroc_vit_large_mae(pretrained=False, **kwargs) -> LROCMAE:
    return _create_lroc("lroc_vit_large", pretrained, encoder_only=False, **kwargs)


@register_full_model
def lroc_vit_huge_mae(pretrained=False, **kwargs) -> LROCMAE:
    return _create_lroc("lroc_vit_huge", pretrained, encoder_only=False, **kwargs)
