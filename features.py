import argparse
import os
import pickle
import time
import math

import h5py
import faiss
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import models
from util import AverageMeter

parser = argparse.ArgumentParser(description='Generate features for deep metric learning')

parser.add_argument('data', metavar='DIR', help='path to matlab dataset matrix')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16', 'inceptionv1'], default='alexnet',
                    help='CNN architecture (default: alexnet)')

parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--weights', type=str, default='', help='path to model weights')
parser.add_argument('--verbose', action='store_true', help='chatty')

def main():
    global args
    args = parser.parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))

    sobel = False
    if args.arch == 'inceptionv1':
      model = models.__dict__[args.arch](sobel=sobel, weight_file='/home/farbod/honours/convert/inception1/kit_pytorch.npy')
    else:
      model = models.__dict__[args.arch](sobel=sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    if args.arch == 'inceptionv1':
      for key in model.modules():
        if isinstance(key, nn.Module): continue
        key = torch.nn.DataParallel(key).cuda()
    else:
      model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading weights '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded weights '{}' (epoch {})"
                  .format(args.weights, checkpoint['epoch']))
        else:
            print("=> no weights found at '{}'".format(args.weights))

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    tra = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize])

    #images = scipy.io.loadmat(args.data)
    with h5py.File(args.data, 'r') as f:
      raw = np.array(f['IMAGES'])
      images = np.swapaxes(raw, 1, 3)

    # load the data
    end = time.time()
    if args.verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))

    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.eval()

    # compute features
    batch_time = AverageMeter()
    end = time.time()

    num_imgs = images.shape[0]
    batch_size = 32
    print num_imgs
    iters = int(math.ceil(num_imgs / batch_size))
    for i in range(iters):
        if i < iters - 1:
          batch = images[i * batch_size: (i + 1) * batch_size]
        else:
          batch = images[i * batch_size:]
        input_tensor = torch.stack([tra(img) for img in batch])
        # discard the label information in the dataloader
        with torch.no_grad():
          input_var = torch.autograd.Variable(input_tensor.cuda())
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((num_imgs, aux.shape[1])).astype('float32')

        if i < iters - 1:
            features[i * batch_size: (i + 1) * batch_size] = aux.astype('float32')
        else:
            # special treatment for final batch
            features[i * batch_size:] = aux.astype('float32')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('{0} / {1}\t'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
              .format(i, iters, batch_time=batch_time))
    features = preprocess_features(features, pca=64)
    scipy.io.savemat('./out3.mat', { 'feats': features })

def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


if __name__ == '__main__':
    main()
