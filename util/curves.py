"""
Training curves from <output_dir>/log.txt (one JSON line per epoch, written by main_pretrain.py).

save_curves(output_dir) writes, next to log.txt:
  log.csv               epoch, lr, train_loss, val_loss and the learned metadata weights, one row per epoch
  training_curves.png   learning rate by epoch, and training/validation loss by epoch

main_pretrain.py calls it after every epoch. To redo it for any run folder:
    python -m util.curves runs/vitb
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

TRAIN, VAL = "#2a78d6", "#eb6834"                 # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_log(output_dir):
    """Rows of log.txt, one per epoch; after a resume, the latest row for an epoch wins."""
    rows = {}
    with open(os.path.join(output_dir, "log.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["epoch"])] = r
    return [rows[e] for e in sorted(rows)]


def write_csv(rows, path):
    first = ["epoch", "lr", "train_loss", "val_loss"]
    extra = sorted({k for r in rows for k in r} - {"epoch", "train_lr", "train_loss", "val_loss"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(first + extra)
        for r in rows:
            w.writerow([r["epoch"], r.get("train_lr"), r.get("train_loss"), r.get("val_loss")]
                       + [r.get(k) for k in extra])


def _style(ax, ylabel):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def plot(rows, path, title=""):
    ep = [r["epoch"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")

    a1.plot(ep, [r.get("train_lr") for r in rows], color=TRAIN, linewidth=2)
    _style(a1, "Mean learning rate in epoch")
    a1.set_title("Learning rate", loc="left", color=INK, fontsize=11)
    a1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    tr, va = [r.get("train_loss") for r in rows], [r.get("val_loss") for r in rows]
    a2.plot(ep, tr, color=TRAIN, linewidth=2, label="Training")
    a2.plot(ep, va, color=VAL, linewidth=2, label="Validation")
    if len(ep) == 1:                                             # a line needs two points
        a2.scatter(ep, tr, color=TRAIN, s=36, zorder=3)
        a2.scatter(ep, va, color=VAL, s=36, zorder=3)
    if all(v and v > 0 for v in tr + va):
        a2.set_yscale("log")
        a2.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a2.yaxis.set_minor_formatter(NullFormatter())
    _style(a2, "Loss (log scale)" if a2.get_yscale() == "log" else "Loss")
    a2.set_title("Loss", loc="left", color=INK, fontsize=11)
    a2.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    a2.margins(x=0.22)
    fig.canvas.draw()                                            # direct labels at the last epoch, kept apart
    labels = [(y, n) for y, n in ((tr[-1], "Training"), (va[-1], "Validation")) if y is not None]
    pt = fig.dpi / 72.0                                          # pixels per point
    line = 8 * 1.4 * pt                                          # one 8-pt text line, in pixels
    pix = sorted((a2.transData.transform((ep[-1], y))[1], y, n) for y, n in labels)
    if len(pix) == 2 and pix[1][0] - pix[0][0] < line:          # closer than one text line: spread them
        mid = (pix[0][0] + pix[1][0]) / 2
        pix = [(mid - line / 2, *pix[0][1:]), (mid + line / 2, *pix[1][1:])]
    for py, y, n in pix:
        dy = (py - a2.transData.transform((ep[-1], y))[1]) / pt    # offset in points
        a2.annotate(f"{n} {y:.4f}", (ep[-1], y), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=8, color=INK)

    for ax in (a1, a2):                                          # whole epochs, only within the data
        ticks = [t for t in MaxNLocator(nbins=8, integer=True).tick_values(ep[0], ep[-1]) if ep[0] <= t <= ep[-1]]
        ax.set_xticks(ticks)

    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_curves(output_dir):
    rows = read_log(output_dir)
    if not rows:
        return
    write_csv(rows, os.path.join(output_dir, "log.csv"))
    plot(rows, os.path.join(output_dir, "training_curves.png"), title=os.path.basename(os.path.normpath(output_dir)))


if __name__ == "__main__":
    for d in sys.argv[1:] or ["."]:
        save_curves(d)
        print(f"wrote {os.path.join(d, 'log.csv')} and {os.path.join(d, 'training_curves.png')}")