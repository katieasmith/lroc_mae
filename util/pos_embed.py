import numpy as np
import torch


# ---- MAE (util/pos_embed.py), unchanged ---------------------------------------------
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega                       # (D/2,)
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    out = np.einsum('m,d->md', pos, omega)          # (M, D/2) outer product
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---- Prithvi-EO-2.0: 3D version (w: 6/16, h: 6/16, t: 4/16 of D) -------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    assert embed_dim % 16 == 0
    t_size, h_size, w_size = grid_size
    w_dim, h_dim, t_dim = embed_dim // 16 * 6, embed_dim // 16 * 6, embed_dim // 16 * 4
    w = get_1d_sincos_pos_embed_from_grid(w_dim, np.arange(w_size))
    h = get_1d_sincos_pos_embed_from_grid(h_dim, np.arange(h_size))
    t = get_1d_sincos_pos_embed_from_grid(t_dim, np.arange(t_size))
    w = np.tile(w, (t_size * h_size, 1))
    h = np.tile(np.repeat(h, w_size, axis=0), (t_size, 1))
    t = np.repeat(t, h_size * w_size, axis=0)
    pos = np.concatenate([w, h, t], axis=1)                       # (t*h*w, D), order (t, h, w)
    if cls_token:
        pos = np.concatenate([np.zeros([1, embed_dim]), pos], axis=0)
    return pos


# ---- Metadata encodings (torch, run every forward pass) --------------------------------
def sincos_1d(values, dim):
    """Prithvi eq. (1) on raw values (degrees, years, days)."""
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, device=values.device, dtype=torch.float32) / (dim / 2.)
    omega = 1. / 10000**omega
    out = values.reshape(-1, 1).float() * omega
    return torch.cat([out.sin(), out.cos()], dim=1)


def sincos_periodic(values_deg, dim):
    """Angles: integer harmonics 1..dim/2, so 0 and 360 encode identically."""
    assert dim % 2 == 0
    k = torch.arange(1, dim // 2 + 1, device=values_deg.device, dtype=torch.float32)
    theta = torch.deg2rad(values_deg.reshape(-1, 1).float()) * k
    return torch.cat([theta.sin(), theta.cos()], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):     # MAE
    """grid_size: int (square, as in MAE) or (h, w). Returns (h*w [+1], D), token order row-major.
    As in MAE, the first D/2 columns encode the column index and the last D/2 the row index."""
    gh, gw = (grid_size, grid_size) if isinstance(grid_size, int) else tuple(grid_size)
    grid_h = np.arange(gh, dtype=np.float32)
    grid_w = np.arange(gw, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, gh, gw])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):                 # MAE
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2), from the w grid
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2), from the h grid
    return np.concatenate([emb_h, emb_w], axis=1)
