# -------------------------------------------------------------------
# LROC MAE pre-training script
# References:
# https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
# -------------------------------------------------------------------

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import models_mae
import util.misc as misc
from engine_pretrain import evaluate, train_one_epoch
from util.config import load_config, metadata_variables
from util.datasets import LROCTiles
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.visualize import fixed_batch, save_reconstructions
from util.tile_store import TileStore
from util.curves import save_curves

def get_args_parser():
    # defaults
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='per GPU')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)                            # Accumulate gradient iterations for increasing the effective batch size under memory constraints

    # model parameters
    parser.add_argument('--model', default='lroc_mae_vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int, help='crop size')        # input image size for the model
    parser.add_argument('--mask_ratio', default=0.75, type=float)                       # percentage of removed patches
    parser.add_argument('--norm_pix_loss', action='store_true')                         # use per-patch normalized pixels as targets for computing loss
    parser.add_argument('--meta_mode', default='patch', choices=['patch', 'global'])    # metadata mode for the model (global is Prithvi's method)

    # optimizer
    parser.add_argument('--weight_decay', type=float, default=0.05)                                             # Prithvi's default
    parser.add_argument('--lr', type=float, default=5e-4, help='absolute lr (Prithvi: 5e-4 at batch 3,840)')    # Prithvi's max learning rate
    # parser.add_argument('--blr', type=float, default=1.5e-4, help='lr = blr * eff_batch / 256 (MAE)')           # Used when lr isn't set, MAE does this
    parser.add_argument('--start_lr', type=float, default=1e-6)                                                 # Prithvi's starting LR
    parser.add_argument('--min_lr', type=float, default=1e-6)                                                   # Prithvi's starting LR
    parser.add_argument('--warmup_epochs', type=int, default=40)                                                # Prithvi's linear warmup epochs
    parser.add_argument('--clip_grad', type=float, default=None)                                                # turn on if there's spikes or NaNs 

    # data: tiles, split and metadata spec live in a YAML file (nested lists don't fit argparse)
    parser.add_argument('--data_config', required=True)                                                        # path to the YAML file containing data configuration
    parser.add_argument('--output_dir', default='./output_dir')                                                # directory for saving checkpoints and logs
    parser.add_argument('--log_dir', default='./output_dir')                                                   # directory for saving logs
    parser.add_argument('--device', default='cuda')                                                            # device to use for training/testing
    parser.add_argument('--seed', default=0, type=int)                                                         # random seed
    parser.add_argument('--resume', default='')                                                                # checkpoint to resume training from
    
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')                                                      # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--save_freq', default=20, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--num_vis', default=4, type=int, help='validation tiles in each reconstruction figure (0=none)')

    # distributed (torchrun sets RANK / WORLD_SIZE / LOCAL_RANK)
    parser.add_argument('--world_size', default=1, type=int)            # number of distributed processes
    parser.add_argument('--local_rank', default=-1, type=int)           # local rank of the process in the distributed setting
    # parser.add_argument('--dist_on_itp', action='store_true')           # use ITP for distributed training
    parser.add_argument('--dist_url', default='env://')                 # URL for setting up distributed training initialization
    return parser

def param_groups_weight_decay(model, weight_decay, skip=("cls_token", "mask_token")):
    """No decay on biases, norms and metadata weights (1-D)"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or name.endswith(".bias") or any(s in name for s in skip) else decay).append(p)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def main(args):
    misc.init_distributed_mode(args)
    print('{}'.format(args).replace(', ', ',\n'))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    cfg = load_config(args.data_config)
    cfg['data']['crop_size'] = args.input_size
    cfg.setdefault('model', {})['patch_size'] = int(args.model.rsplit('patch', 1)[1])
    variables = metadata_variables(cfg)

    Dataset = TileStore if TileStore.available(cfg) else LROCTiles   # extracted tiles if present, else GeoTIFFs
    print(f"reading tiles with {Dataset.__name__}")
    dataset_train = Dataset(cfg, 'train', train=True)
    dataset_val = Dataset(cfg, 'val', train=False)
    
    #dataset_train = LROCTiles(cfg, 'train', train=True)
    #dataset_val = LROCTiles(cfg, 'val', train=False)

    if args.distributed:
        num_tasks, global_rank = misc.get_world_size(), misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_writer = None
    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem,
                     persistent_workers=args.num_workers > 0)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, drop_last=True, **loader_kw)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, drop_last=False, **loader_kw)

    model = models_mae.__dict__[args.model](img_size=args.input_size, in_chans=len(cfg['data'].get('bands', [1])),
                                            norm_pix_loss=args.norm_pix_loss, variables=variables,
                                            meta_mode=args.meta_mode)
    model.to(device)
    print(f"parameters: encoder {sum(p.numel() for p in model.encoder.parameters())/1e6:.2f}M, "
          f"total {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("actual lr: %.2e, effective batch size: %d" % (args.lr, eff_batch_size))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu] if device.type == 'cuda' else None)
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(param_groups_weight_decay(model_without_ddp, args.weight_decay),
                                  lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(enabled=device.type == 'cuda' and args.amp_dtype == torch.float16)
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    vis_batch = None
    if args.output_dir and args.num_vis > 0 and misc.is_main_process():
        vis_batch = fixed_batch(Dataset(cfg, 'val', train=False), args.num_vis)
        if args.start_epoch == 0:
            save_reconstructions(model_without_ddp, vis_batch, device, args,
                                 os.path.join(args.output_dir, "recon-init.png"), log_writer, step=-1,
                                 title="before training")
    
    print(f"Start training for {args.epochs} epochs")
    start_time, best = time.time(), float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler,
                                      log_writer=log_writer, args=args)
        val_stats = evaluate(model, data_loader_val, device, args)
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch)
        if args.output_dir and val_stats['loss'] < best:
            best = val_stats['loss']
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, epoch_name='best')

        if vis_batch is not None:
            save_reconstructions(model_without_ddp, vis_batch, device, args,
                                 os.path.join(args.output_dir, f"recon-{epoch}.png"), log_writer, step=epoch,
                                 title=f"epoch {epoch}  |  val loss {val_stats['loss']:.4f}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'val_loss': val_stats['loss'],
                     **model_without_ddp.metadata_weights(), 'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.add_scalar('val_loss', val_stats['loss'], epoch)
                for k, v in model_without_ddp.metadata_weights().items():
                    log_writer.add_scalar(k, v, epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_curves(args.output_dir)                     # log.csv and training_curves.png

    print('Training time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
