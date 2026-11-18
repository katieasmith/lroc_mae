import math
import sys
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc


def _forward(model, batch, device, args, generator=None):
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.autocast(device_type=device.type, dtype=args.amp_dtype, enabled=device.type == "cuda"):
        return model(b["image"], b["valid"], b["meta"], b["meta_avail"],
                     mask_ratio=args.mask_ratio, generator=generator)


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        # per-iteration (not per-epoch) lr schedule, as in MAE
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        loss, _, _ = _forward(model, batch, device, args)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    """Not in MAE. Validation loss with the same masks every epoch (fixed generator)."""
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(1234)
    metric_logger = misc.MetricLogger(delimiter="  ")
    for batch in data_loader:
        loss, _, _ = _forward(model, batch, device, args, generator=g)
        metric_logger.update(loss=loss.item())
    metric_logger.synchronize_between_processes()
    return {"loss": metric_logger.loss.global_avg}
