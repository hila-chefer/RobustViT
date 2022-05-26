import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from segmentation_dataset import SegmentationDataset, VAL_PARTITION, TRAIN_PARTITION

# Uncomment the expected model below

# ViT
from ViT.ViT import vit_base_patch16_224 as vit
# from ViT.ViT import vit_large_patch16_224 as vit

# ViT-AugReg
# from ViT.ViT_new import vit_small_patch16_224 as vit
# from ViT.ViT_new import vit_base_patch16_224 as vit
# from ViT.ViT_new import vit_large_patch16_224 as vit

# DeiT
# from ViT.ViT import deit_base_patch16_224 as vit
# from ViT.ViT import deit_small_patch16_224 as vit

from ViT.explainer import generate_relevance, get_image_with_relevance
import torchvision
import cv2
from torch.utils.tensorboard import SummaryWriter
import json

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append("vit")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DATA',
                    help='path to dataset')
parser.add_argument('--seg_data', metavar='SEG_DATA',
                    help='path to segmentation dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=3e-6, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--save_interval', default=20, type=int,
                    help='interval to save segmentation results.')
parser.add_argument('--num_samples', default=3, type=int,
                    help='number of samples per class for training')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--lambda_seg', default=0.8, type=float,
                    help='influence of segmentation loss.')
parser.add_argument('--lambda_acc', default=0.2, type=float,
                    help='influence of accuracy loss.')
parser.add_argument('--experiment_folder', default=None, type=str,
                    help='path to folder to use for experiment.')
parser.add_argument('--dilation', default=0, type=float,
                    help='Use dilation on the segmentation maps.')
parser.add_argument('--lambda_background', default=2, type=float,
                    help='coefficient of loss for segmentation background.')
parser.add_argument('--lambda_foreground', default=0.3, type=float,
                    help='coefficient of loss for segmentation foreground.')
parser.add_argument('--num_classes', default=500, type=int,
                    help='coefficient of loss for segmentation foreground.')
parser.add_argument('--temperature', default=1, type=float,
                    help='temperature for softmax (mostly for DeiT).')

best_loss = float('inf')

def main():
    args = parser.parse_args()

    if args.experiment_folder is None:
        args.experiment_folder = f'experiment/' \
                                 f'lr_{args.lr}_seg_{args.lambda_seg}_acc_{args.lambda_acc}' \
                                 f'_bckg_{args.lambda_background}_fgd_{args.lambda_foreground}'
        if args.temperature != 1:
            args.experiment_folder = args.experiment_folder + f'_tempera_{args.temperature}'
        if args.batch_size != 8:
            args.experiment_folder = args.experiment_folder + f'_bs_{args.batch_size}'
        if args.num_classes != 500:
            args.experiment_folder = args.experiment_folder + f'_num_classes_{args.num_classes}'
        if args.num_samples != 3:
            args.experiment_folder = args.experiment_folder + f'_num_samples_{args.num_samples}'
        if args.epochs != 150:
            args.experiment_folder = args.experiment_folder + f'_num_epochs_{args.epochs}'

    if os.path.exists(args.experiment_folder):
        raise Exception(f"Experiment path {args.experiment_folder} already exists!")
    os.mkdir(args.experiment_folder)
    os.mkdir(f'{args.experiment_folder}/train_samples')
    os.mkdir(f'{args.experiment_folder}/val_samples')

    with open(f'{args.experiment_folder}/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        #model = models.__dict__[args.arch]()
        model = vit(pretrained=True).cuda()
        model.train()
        print("done")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print("start")
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
                best_loss = best_loss.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_dataset = SegmentationDataset(args.seg_data, args.data, partition=TRAIN_PARTITION, train_classes=args.num_classes,
                                        num_samples=args.num_samples)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = SegmentationDataset(args.seg_data, args.data, partition=VAL_PARTITION, train_classes=args.num_classes,
                                      num_samples=1)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=10, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        log_dir = os.path.join(args.experiment_folder, 'logs')
        logger = SummaryWriter(log_dir=log_dir)
        args.logger = logger

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        loss1 = validate(val_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = loss1 <= best_loss
        best_loss = min(loss1, best_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, folder=args.experiment_folder)


def train(train_loader, model, criterion, optimizer, epoch, args):
    mse_criterion = torch.nn.MSELoss(reduction='mean')

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    orig_top1 = AverageMeter('Acc@1_orig', ':6.2f')
    orig_top5 = AverageMeter('Acc@5_orig', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5, orig_top1, orig_top5],
        prefix="Epoch: [{}]".format(epoch))

    orig_model = vit(pretrained=True).cuda()
    orig_model.eval()

    # switch to train mode
    model.train()

    for i, (seg_map, image_ten, class_name) in enumerate(train_loader):
        if torch.cuda.is_available():
            image_ten = image_ten.cuda(args.gpu, non_blocking=True)
            seg_map = seg_map.cuda(args.gpu, non_blocking=True)
            class_name = class_name.cuda(args.gpu, non_blocking=True)

        # segmentation loss
        relevance = generate_relevance(model, image_ten, index=class_name)

        reverse_seg_map = seg_map.clone()
        reverse_seg_map[reverse_seg_map == 1] = -1
        reverse_seg_map[reverse_seg_map == 0] = 1
        reverse_seg_map[reverse_seg_map == -1] = 0
        background_loss = mse_criterion(relevance * reverse_seg_map, torch.zeros_like(relevance))
        foreground_loss = mse_criterion(relevance * seg_map, seg_map)
        segmentation_loss = args.lambda_background * background_loss
        segmentation_loss += args.lambda_foreground * foreground_loss

        # classification loss
        output = model(image_ten)
        with torch.no_grad():
            output_orig = orig_model(image_ten)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.flatten()

        if args.temperature != 1:
            output = output / args.temperature
        classification_loss = criterion(output, class_name.flatten())

        loss = args.lambda_seg * segmentation_loss + args.lambda_acc * classification_loss

        # debugging output
        if i % args.save_interval == 0:
            orig_relevance = generate_relevance(orig_model, image_ten, index=class_name)
            for j in range(image_ten.shape[0]):
                image = get_image_with_relevance(image_ten[j], torch.ones_like(image_ten[j]))
                new_vis = get_image_with_relevance(image_ten[j], relevance[j])
                old_vis = get_image_with_relevance(image_ten[j], orig_relevance[j])
                gt = get_image_with_relevance(image_ten[j], seg_map[j])
                h_img = cv2.hconcat([image, gt, old_vis, new_vis])
                cv2.imwrite(f'{args.experiment_folder}/train_samples/res_{i}_{j}.jpg', h_img)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, class_name, topk=(1, 5))
        losses.update(loss.item(), image_ten.size(0))
        top1.update(acc1[0], image_ten.size(0))
        top5.update(acc5[0], image_ten.size(0))

        # metrics for original vit
        acc1_orig, acc5_orig = accuracy(output_orig, class_name, topk=(1, 5))
        orig_top1.update(acc1_orig[0], image_ten.size(0))
        orig_top5.update(acc5_orig[0], image_ten.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)
            args.logger.add_scalar('{}/{}'.format('train', 'segmentation_loss'), segmentation_loss,
                                   epoch*len(train_loader)+i)
            args.logger.add_scalar('{}/{}'.format('train', 'classification_loss'), classification_loss,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'orig_top1'), acc1_orig,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'top1'), acc1,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'orig_top5'), acc5_orig,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'top5'), acc5,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'tot_loss'), loss,
                                   epoch * len(train_loader) + i)


def validate(val_loader, model, criterion, epoch, args):
    mse_criterion = torch.nn.MSELoss(reduction='mean')

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    orig_top1 = AverageMeter('Acc@1_orig', ':6.2f')
    orig_top5 = AverageMeter('Acc@5_orig', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5, orig_top1, orig_top5],
        prefix="Epoch: [{}]".format(val_loader))

    # switch to evaluate mode
    model.eval()

    orig_model = vit(pretrained=True).cuda()
    orig_model.eval()

    with torch.no_grad():
        for i, (seg_map, image_ten, class_name) in enumerate(val_loader):
            if args.gpu is not None:
                image_ten = image_ten.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                seg_map = seg_map.cuda(args.gpu, non_blocking=True)
                class_name = class_name.cuda(args.gpu, non_blocking=True)

                # segmentation loss
                with torch.enable_grad():
                    relevance = generate_relevance(model, image_ten, index=class_name)

                reverse_seg_map = seg_map.clone()
                reverse_seg_map[reverse_seg_map == 1] = -1
                reverse_seg_map[reverse_seg_map == 0] = 1
                reverse_seg_map[reverse_seg_map == -1] = 0
                background_loss = mse_criterion(relevance * reverse_seg_map, torch.zeros_like(relevance))
                foreground_loss = mse_criterion(relevance * seg_map, seg_map)
                segmentation_loss = args.lambda_background * background_loss
                segmentation_loss += args.lambda_foreground * foreground_loss

                # classification loss
                with torch.no_grad():
                    output = model(image_ten)
                    output_orig = orig_model(image_ten)

                _, pred = output.topk(1, 1, True, True)
                pred = pred.flatten()
                if args.temperature != 1:
                    output = output / args.temperature
                classification_loss = criterion(output, class_name.flatten())

                loss = args.lambda_seg * segmentation_loss + args.lambda_acc * classification_loss

            # save results
            if i % args.save_interval == 0:
                with torch.enable_grad():
                    orig_relevance = generate_relevance(orig_model, image_ten, index=class_name)
                for j in range(image_ten.shape[0]):
                    image = get_image_with_relevance(image_ten[j], torch.ones_like(image_ten[j]))
                    new_vis = get_image_with_relevance(image_ten[j], relevance[j])
                    old_vis = get_image_with_relevance(image_ten[j], orig_relevance[j])
                    gt = get_image_with_relevance(image_ten[j], seg_map[j])
                    h_img = cv2.hconcat([image, gt, old_vis, new_vis])
                    cv2.imwrite(f'{args.experiment_folder}/val_samples/res_{i}_{j}.jpg', h_img)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, class_name, topk=(1, 5))
            losses.update(loss.item(), image_ten.size(0))
            top1.update(acc1[0], image_ten.size(0))
            top5.update(acc5[0], image_ten.size(0))

            # metrics for original vit
            acc1_orig, acc5_orig = accuracy(output_orig, class_name, topk=(1, 5))
            orig_top1.update(acc1_orig[0], image_ten.size(0))
            orig_top5.update(acc5_orig[0], image_ten.size(0))

            if i % args.print_freq == 0:
                progress.display(i)
                args.logger.add_scalar('{}/{}'.format('val', 'segmentation_loss'), segmentation_loss,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'classification_loss'), classification_loss,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'orig_top1'), acc1_orig,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'top1'), acc1,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'orig_top5'), acc5_orig,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'top5'), acc5,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'tot_loss'), loss,
                                       epoch * len(val_loader) + i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg


def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    torch.save(state, f'{folder}/{filename}')
    if is_best:
        shutil.copyfile(f'{folder}/{filename}', f'{folder}/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.85 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()