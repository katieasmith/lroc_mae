"""
MAE-style entry point (facebookresearch/mae models_mae.py): main_pretrain.py builds the model with
models_mae.__dict__[args.model](...). The classes live in models/lroc_mae.py, configs in models/lroc_vit.py.
"""

from models.lroc_mae import LROCDecoder, LROCMAE, LROCViT, MetadataEncoder, random_masking  # noqa: F401
from models.lroc_vit import _create_lroc


def lroc_mae_vit_tiny_patch16(**kwargs):    # debugging
    return _create_lroc("lroc_vit_tiny", encoder_only=False, **kwargs)


def lroc_mae_vit_base_patch16(**kwargs):    # MAE mae_vit_base_patch16_dec512d8b
    return _create_lroc("lroc_vit_base", encoder_only=False, **kwargs)


def lroc_mae_vit_large_patch16(**kwargs):   # Prithvi-EO-2.0-300M
    return _create_lroc("lroc_vit_large", encoder_only=False, **kwargs)


def lroc_mae_vit_huge_patch14(**kwargs):    # Prithvi-EO-2.0-600M
    return _create_lroc("lroc_vit_huge", encoder_only=False, **kwargs)
