"""
LROC FM masked autoencoder: ViT encoder, MAE decoder and the MAE wrapper, for single images (B, C, H, W)
with MAE's fixed 2D sin/cos positional embedding.

Structure follows terratorch/models/backbones/prithvi_mae.py (PatchEmbed, PrithviViT, MAEDecoder,
PrithviMAE; IBM, Apache-2.0), which restructures facebookresearch/mae models_mae.py
(MaskedAutoencoderViT). Changes for LROC:
  * MetadataEncoder replaces Prithvi's TemporalEncoder/LocationEncoder: any number of groups, values
    per patch ("patch") or per sample ("global" = Prithvi). Added in the encoder before masking and in
    the decoder after unshuffling, as in Prithvi.
  * forward_loss ignores pixels flagged invalid (nodata / ISIS special pixels).
  * No random metadata drop: groups are left out only when the data marks them unavailable (meta_avail).
  * random_masking accepts a generator, so validation can use the same masks every epoch.
  * 2D instead of Prithvi's 3D: Conv2d patch embedding and MAE's 2D sin/cos positional embedding (all D
    dimensions encode row/column). Inputs with a time axis of length 1, (B, C, 1, H, W), are accepted.
"""

from __future__ import annotations

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from util.pos_embed import get_2d_sincos_pos_embed, sincos_1d, sincos_periodic

LayerNorm = partial(nn.LayerNorm, eps=1e-6)        # MAE's factories; Prithvi's class default is eps=1e-5


# ---------------------------------------------------------------------------------------------
# Helpers (prithvi_mae.py / models_mae.py)
# ---------------------------------------------------------------------------------------------
def _init_weights(module):
    """Xavier-uniform for Linear, unit/zero LayerNorm (MAE and Prithvi)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def random_masking(sequence, mask_ratio, noise=None, generator=None):
    """Per-sample random masking by argsort of uniform noise (MAE / Prithvi).
    Returns the kept tokens, the binary mask (1 = masked) and the indices that undo the shuffle."""
    batch_size, seq_length, dim = sequence.shape
    len_keep = int(seq_length * (1 - mask_ratio))
    if noise is None:
        noise = torch.rand(batch_size, seq_length, device=sequence.device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)                    # small = keep, large = remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
    mask = torch.ones([batch_size, seq_length], device=sequence.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return sequence_unmasked, mask, ids_restore


def _pos_embed_tensor(embed_dim, grid_size):
    """(1, 1 + h*w, D): a zero row for the cls token, then MAE's 2D sin/cos embedding."""
    return torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, tuple(grid_size), cls_token=True)).float().unsqueeze(0)


def _interpolate_pos_encoding(pos_embed, grid_size, patch_size, shape, embed_dim):
    """Prithvi's resizing of the fixed positional embedding (2D case): bicubic interpolation of the
    patch embeddings when the input height/width differs from the size the model was built for."""
    h_patches, w_patches = shape[-2] // patch_size[0], shape[-1] // patch_size[1]
    if [h_patches, w_patches] == list(grid_size):
        return pos_embed
    class_pos_embed, patch_pos_embed = pos_embed[:, :1], pos_embed[:, 1:]
    patch_pos_embed = patch_pos_embed.reshape(1, *grid_size, embed_dim).permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(h_patches, w_patches),
                                                mode="bicubic", align_corners=True)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


def _to_2d(x):
    """(B, C, 1, H, W) -> (B, C, H, W); 4D tensors pass through."""
    if x is not None and x.dim() == 5:
        assert x.shape[2] == 1, "the 2D model takes one image per sample (time axis of length 1)"
        x = x[:, :, 0]
    return x


def _pair(patch_size):
    """16 -> (16, 16); also accepts (16, 16) and Prithvi-style (1, 16, 16)."""
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    patch_size = tuple(patch_size)
    if len(patch_size) == 3:
        assert patch_size[0] == 1, "temporal patches need the 3D model"
        patch_size = patch_size[1:]
    return patch_size


# ---------------------------------------------------------------------------------------------
# Patch embedding (timm PatchEmbed, as in MAE; Conv2d)
# ---------------------------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, input_size=(224, 224), patch_size=(16, 16), in_chans=1, embed_dim=768,
                 norm_layer=None, flatten=True, bias=True):
        super().__init__()
        self.input_size, self.patch_size = tuple(input_size), tuple(patch_size)
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        assert self.grid_size >= [1, 1], "Patch size is bigger than input size."
        self.num_patches = math.prod(self.grid_size)
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):                                      # (B, C, H, W)
        if any(s % p for s, p in zip(x.shape[-2:], self.patch_size)):
            warnings.warn(f"Input {tuple(x.shape[-2:])} is not divisible by patch size {self.patch_size}; "
                          "the border is ignored.")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)                    # (B, D, h, w) -> (B, L, D), row-major
        return self.norm(x)


# ---------------------------------------------------------------------------------------------
# Metadata (replaces prithvi_mae.TemporalEncoder / LocationEncoder)
# ---------------------------------------------------------------------------------------------
class MetadataEncoder(nn.Module):
    """meta (B, V, h, w) -> (B, h*w, D). Each variable is encoded independently with 1D sin/cos
    (integer harmonics if periodic), a group's encodings are concatenated to D, multiplied by one
    learned weight per group (init 0.1, as Prithvi's coords_scale_learn) and the groups are summed."""

    def __init__(self, embed_dim, variables, init_scale=0.1, mode="patch"):
        super().__init__()
        assert mode in ("patch", "global")
        self.mode, self.dim = mode, embed_dim
        self.groups = list(dict.fromkeys(v["group"] for v in variables))
        self.idx = [[i for i, v in enumerate(variables) if v["group"] == g] for g in self.groups]
        self.periodic = [bool(v.get("periodic", False)) for v in variables]
        self.dims = []
        for idx in self.idx:
            base = (embed_dim // len(idx)) // 2 * 2
            self.dims.append([base] * (len(idx) - 1) + [embed_dim - base * (len(idx) - 1)])
        self.scale = nn.Parameter(torch.full((len(self.groups),), float(init_scale)))

    def forward(self, meta, keep=None):
        B, V, h, w = meta.shape
        if self.mode == "global":                               # one value per sample, as Prithvi
            rad = torch.deg2rad(meta)
            circ = torch.rad2deg(torch.atan2(rad.sin().mean((2, 3)), rad.cos().mean((2, 3))))
            per = torch.tensor(self.periodic, device=meta.device)[None]
            meta = torch.where(per, circ, meta.mean((2, 3)))[..., None, None]
        L = meta.shape[2] * meta.shape[3]
        out = 0
        for g, (idx, dims) in enumerate(zip(self.idx, self.dims)):
            parts = [(sincos_periodic if self.periodic[v] else sincos_1d)(meta[:, v], d).view(B, L, d)
                     for v, d in zip(idx, dims)]
            emb = torch.cat(parts, dim=-1) * self.scale[g]
            if keep is not None:
                emb = emb * keep[:, g].view(B, 1, 1).to(emb.dtype)
            out = out + emb
        return out.expand(B, h * w, self.dim) if self.mode == "global" else out


def _add_metadata(x, meta_encoder, meta, keep):
    """Add the metadata embedding to (B, L, D) tokens; meta is (B, V, h, w) on the patch grid."""
    if meta_encoder is None or meta is None:
        return x
    assert x.shape[1] == meta.shape[-2] * meta.shape[-1], \
        f"metadata grid {tuple(meta.shape[-2:])} does not match {x.shape[1]} tokens"
    return x + meta_encoder(meta, keep)


# ---------------------------------------------------------------------------------------------
# Encoder (prithvi_mae.PrithviViT / models_mae.forward_encoder)
# ---------------------------------------------------------------------------------------------
class LROCViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, norm_layer=LayerNorm, variables=None, meta_mode="patch", init_scale=0.1,
                 drop_path=0.0, num_frames=1, **kwargs):
        super().__init__()
        assert num_frames == 1, "the 2D model takes one image per sample"
        self.in_chans, self.embed_dim = in_chans, embed_dim
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.patch_embed = PatchEmbed(self.img_size, _pair(patch_size), in_chans, embed_dim)
        self.out_channels = [embed_dim] * depth
        self.variables = list(variables or [])
        self.meta = MetadataEncoder(embed_dim, self.variables, init_scale, meta_mode) if self.variables else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                                           drop_path=drop_path) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        self.pos_embed.data.copy_(_pos_embed_tensor(self.embed_dim, self.patch_embed.grid_size))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))         # like nn.Linear (MAE)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def random_masking(self, sequence, mask_ratio, noise=None, generator=None):
        return random_masking(sequence, mask_ratio, noise, generator)

    def interpolate_pos_encoding(self, sample_shape):
        return _interpolate_pos_encoding(self.pos_embed, self.patch_embed.grid_size, self.patch_embed.patch_size,
                                         sample_shape, self.embed_dim)

    def _embed(self, x, meta=None, keep=None):
        x = _to_2d(x)
        pos_embed = self.interpolate_pos_encoding(x.shape[-2:])
        x = self.patch_embed(x)
        x = x + pos_embed[:, 1:, :]
        x = _add_metadata(x, self.meta, meta, keep)
        return x, pos_embed

    def _with_cls(self, x, pos_embed):
        cls_token = (self.cls_token + pos_embed[:, :1, :]).to(x.dtype)
        return torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

    def forward(self, x, meta=None, keep=None, mask_ratio=0.75, generator=None):
        x, pos_embed = self._embed(x, meta, keep)
        x, mask, ids_restore = self.random_masking(x, mask_ratio, generator=generator)
        x = self._with_cls(x, pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore

    def forward_features(self, x, meta=None, keep=None, meta_avail=None, **kwargs):
        """No masking; every block's output (B, 1 + L, D), the last one normalised.
        TerraTorch tasks pass every extra batch key (e.g. valid) as a keyword argument; unused ones are ignored."""
        keep = meta_avail.bool() if keep is None and meta_avail is not None else keep
        x, pos_embed = self._embed(x, meta, keep)
        x = self._with_cls(x, pos_embed)
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x.clone())
        out[-1] = self.norm(x)
        return out

    def prepare_features_for_image_model(self, features, grid_hw=None):
        """(B, 1 + h*w, D) -> (B, D, h, w) for convolutional decoders (UPerNet etc.)."""
        out = []
        for x in features:
            x = x[:, 1:, :]
            h, w = grid_hw or (math.isqrt(x.shape[1]),) * 2
            out.append(x.reshape(x.shape[0], h, w, -1).permute(0, 3, 1, 2))
        return out

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)


# ---------------------------------------------------------------------------------------------
# Decoder (prithvi_mae.MAEDecoder / models_mae.forward_decoder)
# ---------------------------------------------------------------------------------------------
class LROCDecoder(nn.Module):
    def __init__(self, patch_size=(16, 16), grid_size=(14, 14), in_chans=1, encoder_embed_dim=768,
                 decoder_embed_dim=512, depth=8, num_heads=16, mlp_ratio=4.0, norm_layer=LayerNorm,
                 variables=None, meta_mode="patch", init_scale=0.1):
        super().__init__()
        self.patch_size, self.grid_size = tuple(patch_size), list(grid_size)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer("decoder_pos_embed", torch.zeros(1, math.prod(grid_size) + 1, decoder_embed_dim))
        variables = list(variables or [])
        self.meta = MetadataEncoder(decoder_embed_dim, variables, init_scale, meta_mode) if variables else None
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                                                   norm_layer=norm_layer) for _ in range(depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, math.prod(self.patch_size) * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        self.decoder_pos_embed.data.copy_(_pos_embed_tensor(self.decoder_embed_dim, self.grid_size))
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def interpolate_pos_encoding(self, sample_shape):
        return _interpolate_pos_encoding(self.decoder_pos_embed, self.grid_size, self.patch_size,
                                         sample_shape, self.decoder_embed_dim)

    def forward(self, hidden_states, ids_restore, meta=None, keep=None, input_size=None):
        x = self.decoder_embed(hidden_states)
        cls_token = x[:, :1, :]
        mask_tokens = self.mask_token.to(x.dtype).repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x[:, 1:, :], mask_tokens], dim=1)       # drop cls, append mask tokens
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        pos_embed = self.interpolate_pos_encoding(input_size[-2:])
        cls_token = cls_token + pos_embed[:, :1, :]
        x = x + pos_embed[:, 1:, :]
        x = _add_metadata(x, self.meta, meta, keep)
        x = torch.cat([cls_token.to(x.dtype), x], dim=1)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)[:, 1:, :]                  # drop cls


# ---------------------------------------------------------------------------------------------
# Masked autoencoder (prithvi_mae.PrithviMAE / models_mae.MaskedAutoencoderViT)
# ---------------------------------------------------------------------------------------------
class LROCMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=LayerNorm,
                 norm_pix_loss=False, variables=None, meta_mode="patch", init_scale=0.1,
                 drop_path=0.0, mask_ratio=0.75, num_frames=1):
        super().__init__()
        self.variables = list(variables or [])
        self.encoder = LROCViT(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, norm_layer,
                               self.variables, meta_mode, init_scale, drop_path, num_frames)
        self.decoder = LROCDecoder(self.encoder.patch_embed.patch_size, self.encoder.patch_embed.grid_size,
                                   in_chans, embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads,
                                   mlp_ratio, norm_layer, self.variables, meta_mode, init_scale)
        self.mask_ratio, self.norm_pix_loss = mask_ratio, norm_pix_loss
        self.n_groups = len(self.encoder.meta.groups) if self.encoder.meta else 0

    def patchify(self, imgs):                                  # MAE, any C: (B, C, H, W) -> (B, L, p*q*C)
        p, q = self.encoder.patch_embed.patch_size
        B, C, H, W = imgs.shape
        x = imgs.reshape(B, C, H // p, p, W // q, q)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(B, (H // p) * (W // q), p * q * C)

    def unpatchify(self, x, shape):                            # inverse; shape = (C, H, W)
        p, q = self.encoder.patch_embed.patch_size
        C, H, W = shape
        x = x.reshape(x.shape[0], H // p, W // q, p, q, C)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], C, H, W)

    def forward_loss(self, pixel_values, pred, mask, valid=None):
        """MSE on masked patches; invalid pixels excluded (MAE/Prithvi when everything is valid)."""
        target = self.patchify(pixel_values)
        v = (torch.ones_like(target) if valid is None
             else self.patchify(valid.expand_as(pixel_values).to(target.dtype)))
        if self.norm_pix_loss:
            n = v.sum(-1, keepdim=True).clamp(min=1)
            mean = (target * v).sum(-1, keepdim=True) / n
            var = (((target - mean) ** 2) * v).sum(-1, keepdim=True) / (n - 1).clamp(min=1)  # unbiased, as torch.var
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = ((pred.float() - target) ** 2) * v
        loss = loss.sum(-1) / v.sum(-1).clamp(min=1)            # mean per patch over valid pixels
        w = mask * (v.sum(-1) > 0)                              # masked patches that contain data
        return (loss * w).sum() / w.sum().clamp(min=1)

    def _keep(self, meta, meta_avail):
        """Groups to use per sample: all of them, unless the data marks a group unavailable."""
        if meta is None or not self.n_groups or meta_avail is None:
            return None
        return meta_avail.bool()

    def forward(self, pixel_values, valid=None, meta=None, meta_avail=None, mask_ratio=None, generator=None):
        pixel_values, valid = _to_2d(pixel_values), _to_2d(valid)
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        keep = self._keep(meta, meta_avail)
        latent, mask, ids_restore = self.encoder(pixel_values, meta, keep, mask_ratio, generator)
        pred = self.decoder(latent, ids_restore, meta, keep, input_size=pixel_values.shape)
        loss = self.forward_loss(pixel_values, pred, mask, valid)
        return loss, pred, mask

    def forward_features(self, x, meta=None, keep=None):
        return self.encoder.forward_features(x, meta, keep)

    def metadata_weights(self):
        out = {}
        for tag, enc in (("enc", self.encoder.meta), ("dec", self.decoder.meta)):
            if enc is not None:
                out.update({f"w_{tag}_{g}": s for g, s in zip(enc.groups, enc.scale.detach().cpu().tolist())})
        return out
