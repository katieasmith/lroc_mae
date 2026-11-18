"""
Reconstruction figures during pretraining: ground truth, what the model sees (visible patches), the
model's output, and the output with the visible patches pasted back (as in the MAE paper's figures).
"""

import matplotlib

matplotlib.use("Agg")                                   # no display on the server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

def fixed_batch(dataset, n=4):
    """n samples spread evenly over the validation set (centre crop, no flip), the same every time."""
    idx = np.linspace(0, len(dataset) - 1, n).round().astype(int)
    return default_collate([dataset[int(i)] for i in idx])


@torch.no_grad()
def save_reconstructions(model, batch, device, args, path, log_writer=None, step=None, title=""):
    """model: the unwrapped model (model_without_ddp). Masks come from a fixed seed, so every figure in a
    run uses the same masked patches and only the model changes."""
    was_training = model.training
    model.eval()
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    g = torch.Generator(device=device)
    g.manual_seed(0)
    with torch.autocast(device_type=device.type, dtype=args.amp_dtype, enabled=device.type == "cuda"):
        _, pred, mask = model(b["image"], b["valid"], b["meta"], b["meta_avail"],
                              mask_ratio=args.mask_ratio, generator=g)
    model.train(was_training)

    x = b["image"].float()
    x = x[:, :, 0] if x.dim() == 5 else x                  # (B, C, H, W)
    valid = b["valid"].float()
    valid = valid[:, :, 0] if valid.dim() == 5 else valid  # (B, 1, H, W)
    pred = pred.float()
    if model.norm_pix_loss:                               # predictions are per-patch standardised: undo it
        target = model.patchify(x)
        mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
        pred = pred * (var + 1e-6) ** 0.5 + mean
    shape = tuple(x.shape[1:])
    rec = model.unpatchify(pred, shape)
    m = model.unpatchify(mask.unsqueeze(-1).expand(-1, -1, pred.shape[-1]).float(), shape)
    pasted = x * (1 - m) + rec * m
    seen = x.masked_fill(m.bool(), float("nan"))
    invalid = (valid == 0).expand_as(x)

    panels = (("1. Ground truth", x.masked_fill(invalid, float("nan"))),
              ("2. What the model sees", seen.masked_fill(invalid, float("nan"))),
              ("3. Reconstruction", rec),
              ("4. Reconstruction + visible", pasted.masked_fill(invalid, float("nan"))))
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("black")
    n = x.shape[0]
    fig, axes = plt.subplots(n, len(panels), figsize=(3.2 * len(panels), 3.2 * n), squeeze=False)
    for i in range(n):
        ok = x[i, 0][~invalid[i, 0]]
        lo, hi = (torch.quantile(ok, torch.tensor([0.01, 0.99], device=ok.device)).tolist()
                  if ok.numel() else (0.0, 1.0))        # same grey scale for all panels of a sample
        for j, (name, img) in enumerate(panels):
            axes[i, j].imshow(img[i, 0].cpu().numpy(), cmap=cmap, vmin=lo, vmax=hi)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(name)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    if log_writer is not None:
        log_writer.add_figure("reconstruction", fig, global_step=step)
    plt.close(fig)