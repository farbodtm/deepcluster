# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import scipy.io
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering
import entropy_2 as entropy
import models
from util import AverageMeter, Logger, UnifLabelSampler
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16', 'inceptionv1'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--num_classes', '--num_cls', type=int, default=1000,
                    help='number of classes (default: 1000)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    logs = []

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))

    if args.arch == 'inceptionv1':
      model = models.__dict__[args.arch](sobel=args.sobel, weight_file='/home/farbod/honours/convert/inception1/kit_pytorch.npy', out=args.num_classes)
    else:
      model = models.__dict__[args.arch](sobel=args.sobel, out=args.num_classes)

    fd = int(model.top_layer.weight.size()[1])
    if args.arch == 'inceptionv1' or args.arch == 'mnist':
      for key in model.modules():
        if isinstance(key, nn.Module): continue
        key = torch.nn.DataParallel(key).cuda()
    else:
      model.features = torch.nn.DataParallel(model.features)

    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer1 = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )
    optimizer2 = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )
    optimizer2 = optimizer1

    # define loss function
    criterion = entropy.EntropyLoss().cuda()
    #criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            model.top_layer = None
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])

            model.top_layer = nn.Linear(4096, args.num_classes)
            model.top_layer.weight.data.normal_(0, 0.01)
            model.top_layer.bias.data.zero_()
            model.top_layer = model.top_layer.cuda()
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    #for param in model.parameters():
    #  param.requires_grad = False
    #for param in model.classifier.parameters():
    #  param.requires_grad = True
    #for param in model.top_layer.parameters():
    #  param.requires_grad = True

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    plot_dir = os.path.join(args.exp, 'plots')
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    # creating logger
    logger = Logger(os.path.join(args.exp, 'log'))

    # preprocessing of data
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    if args.verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))

    loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True)

    #sampler = UnifLabelSampler(int(len(dataset)),
    #        last_assignment)
    #loader = torch.utils.data.DataLoader(dataset,
    #        batch_size=args.batch,
    #        num_workers=args.workers,
    #        pin_memory=True,
    #        sampler=sampler)


    noshuff_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch/2,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             shuffle=False)

    # get ground truth labels for nmi
    #num_classes = args.num_classes
    num_classes = args.num_classes
    labels = [ [] for i in range(num_classes) ]
    for i, (_, label) in enumerate(dataset.imgs):
      labels[label].append(i)

    last_assignment = None
    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        last_assignment = None
        loss, predicted = train(loader, noshuff_loader, model, criterion, optimizer1, optimizer2, epoch, last_assignment)


        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'ConvNet loss: {2:.3f}'
                  .format(epoch, time.time() - end, loss))
            nmi_prev = 0
            nmi_gt = 0
            try:
                nmi_prev = normalized_mutual_info_score(
                    predicted,
                    logger.data[-1]
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi_prev))
            except IndexError:
                pass

            nmi_gt = normalized_mutual_info_score(
                predicted,
                clustering.arrange_clustering(labels)
            )
            print('NMI against ground-truth labels: {0:.3f}'.format(nmi_gt))
            print('####################### \n')
            logs.append([epoch, loss, nmi_prev, nmi_gt])
        # save running checkpoint
        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save({'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer1' : optimizer1.state_dict(),
                        'optimizer2' : optimizer2.state_dict()},
                       os.path.join(args.exp, 'checkpoint_{}.pth.tar'.format(epoch+1)))
        # save cluster assignments
        logger.log(predicted)
        last_assignment = [[] for i in range(args.num_classes)]
        for i in range(len(predicted)):
            last_assignment[predicted[i]].append(i)
        for i in last_assignment:
            print len(i)

    scipy.io.savemat(os.path.join(args.exp, 'logs.mat'), { 'logs': np.array(logs)})

def train(loader, noshuff_loader, model, crit, opt1, opt2, epoch, last_assignment):
    """Training of the CNN.
        Args:
            dataset (torch.utils.data.Dataset): Data set
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()
    predicted = []

    end = time.time()
    rn = np.random.randint(1000)
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer1' : opt1.state_dict(),
                'optimizer2' : opt2.state_dict()
            }, path)

        input_var = torch.autograd.Variable(input_tensor.cuda())

        output = model(input_var)

        #target_var = torch.autograd.Variable(target.cuda(async=True))
        #loss = crit(output, target_var)

        if (((i+rn)/10)%2 != 0):
            loss = crit(output, total=True)
            print loss

            # record loss
            losses.update(loss.data[0], input_tensor.size(0))

            # compute gradient and do SGD step
            opt1.zero_grad()
            loss.backward()
            opt1.step()
        else:
            loss = crit(output, total=False)
            print loss

            # record loss
            losses.update(loss.data[0], input_tensor.size(0))

            # compute gradient and do SGD step
            opt2.zero_grad()
            loss.backward()
            opt2.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    model.eval()

    for i, (input_tensor, target) in enumerate(noshuff_loader):

        # save checkpoint
        n = len(loader) * epoch + i

        input_var = torch.autograd.Variable(input_tensor.cuda())

        output = model(input_var)
        _, argmax = torch.max(output, 1)
        predicted.append(argmax)
    predicted = torch.cat(predicted)
    return losses.avg, predicted


if __name__ == '__main__':
    main()
