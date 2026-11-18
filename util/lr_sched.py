import math


def adjust_learning_rate(optimizer, epoch, args):
    """MAE's warmup + half-cycle cosine; `epoch` is fractional. Only change: warmup starts at
    args.start_lr (Prithvi: 1e-6) instead of 0."""
    if epoch < args.warmup_epochs:
        lr = args.start_lr + (args.lr - args.start_lr) * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
    return lr
