import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import warnings
import torch.distributed
import numpy as np
import random
import faulthandler
import torch.multiprocessing as mp
import time
import scipy.misc
from models.networks import PointFlow
from torch import optim
from args import get_args
from torch.backends import cudnn
from utils import AverageValueMeter, set_random_seed, apply_random_rotation, save, resume, visualize_point_clouds
from tensorboardX import SummaryWriter
from datasets import ProteinData


def main_worker(gpu, save_dir, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()

    writer = SummaryWriter(logdir=log_dir)

    if not args.use_latent_flow:  # auto-encoder only
        args.prior_weight = 0
        args.entropy_weight = 0

    # load model to GPU
    model = PointFlow(args)
    model.cuda(args.gpu)

    # resume checkpoints
    start_epoch = 0
    optimizer = model.make_optimizer(args)

    # initialize datasets and loaders
    tr_dataset = ProteinData(args.data_dir)
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=True
        num_workers=0, pin_memory=True, drop_last=True
    )

    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"

    # main training loop
    start_time = time.time()
    entropy_avg_meter = AverageValueMeter()
    latent_nats_avg_meter = AverageValueMeter()
    point_nats_avg_meter = AverageValueMeter()

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):

        # adjust the learning rate
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

        # train for one epoch
        for bidx, data in enumerate(train_loader):
            tr_batch = data
            step = bidx + len(train_loader) * epoch
            model.train()
            inputs = tr_batch.cuda(args.gpu, non_blocking=True)
            out = model(inputs, optimizer, step, writer)
            entropy, prior_nats, recon_nats = out['entropy'], out['prior_nats'], out['recon_nats']
            entropy_avg_meter.update(entropy)
            point_nats_avg_meter.update(recon_nats)
            latent_nats_avg_meter.update(prior_nats)

        # save visualizations
        if (epoch + 1) % args.viz_freq == 0:
            # reconstructions
            model.eval()
            samples = model.reconstruct(inputs)
            results = []
            for idx in range(min(10, inputs.size(0))):
                res = visualize_point_clouds(samples[idx], inputs[idx], idx,
                                             pert_order=train_loader.dataset.display_axis_order)
                results.append(res)
            res = np.concatenate(results, axis=1)
            scipy.misc.imsave(os.path.join(save_dir, 'images', 'tr_vis_conditioned_epoch%d-gpu%s.png' % (epoch, args.gpu)),
                              res.transpose((1, 2, 0)))
            if writer is not None:
                writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)

            # samples
            if args.use_latent_flow:
                num_samples = min(10, inputs.size(0))
                num_points = inputs.size(1)
                _, samples = model.sample(num_samples, num_points)
                results = []
                for idx in range(num_samples):
                    res = visualize_point_clouds(samples[idx], inputs[idx], idx,
                                                 pert_order=train_loader.dataset.display_axis_order)
                    results.append(res)
                res = np.concatenate(results, axis=1)
                scipy.misc.imsave(os.path.join(save_dir, 'images', 'tr_vis_conditioned_epoch%d-gpu%s.png' % (epoch, args.gpu)),
                                  res.transpose((1, 2, 0)))
                if writer is not None:
                    writer.add_image('tr_vis/sampled', torch.as_tensor(res), epoch)


def main():
    # command line args
    args = get_args()
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'images'))

    with open(os.path.join(save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.sync_bn:
        assert args.distributed

    main_worker(args.gpu, save_dir, args)


if __name__ == '__main__':
    main()
