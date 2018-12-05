from collections import OrderedDict
import math

import torch
import torch.nn as nn

__all__ = [ 'GoogLeNet', 'inceptionv1']

class Inception(nn.Module):
  def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
    super(Inception, self).__init__()
    # 1x1 conv branch
    self.b1 = nn.Sequential(
        nn.Conv2d(in_planes, n1x1, kernel_size=1),
        nn.BatchNorm2d(n1x1),
        nn.ReLU(True),
        )

    # 1x1 conv -> 3x3 conv branch
    self.b2 = nn.Sequential(
        nn.Conv2d(in_planes, n3x3red, kernel_size=1),
        nn.BatchNorm2d(n3x3red),
        nn.ReLU(True),
        nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
        nn.BatchNorm2d(n3x3),
        nn.ReLU(True),
        )

    # 1x1 conv -> 5x5 conv branch
    self.b3 = nn.Sequential(
        nn.Conv2d(in_planes, n5x5red, kernel_size=1),
        nn.BatchNorm2d(n5x5red),
        nn.ReLU(True),
        nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
        nn.BatchNorm2d(n5x5),
        nn.ReLU(True),
        nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
        nn.BatchNorm2d(n5x5),
        nn.ReLU(True),
        )

    # 3x3 pool -> 1x1 conv branch
    self.b4 = nn.Sequential(
        nn.MaxPool2d(3, stride=1, padding=1),
        nn.Conv2d(in_planes, pool_planes, kernel_size=1),
        nn.BatchNorm2d(pool_planes),
        nn.ReLU(True),
        )

  def forward(self, x):
    y1 = self.b1(x)
    y2 = self.b2(x)
    y3 = self.b3(x)
    y4 = self.b4(x)
    return torch.cat([y1, y2, y3, y4], 1)

class GoogLeNet(nn.Module):

  def __init__(self, features, num_classes, sobel):
    super(GoogLeNet, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(True),
        )
    self.top_layer = nn.Linear(1024, num_classes)

    self._initialize_weights()
    if sobel:
      grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
      grayscale.weight.data.fill_(1.0 / 3.0)
      grayscale.bias.data.zero_()
      sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
      sobel_filter.weight.data[0,0].copy_(
          torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
          )
      sobel_filter.weight.data[1,0].copy_(
          torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
          )
      sobel_filter.bias.data.zero_()
      self.sobel = nn.Sequential(grayscale, sobel_filter)
      for p in self.sobel.parameters():
        p.requires_grad = False
    else:
      self.sobel = None

  def forward(self, x):
    out = self.features['pre1'](x)
    out = self.features['pre2'](out)
    out = self.features['a3'](out)
    out = self.features['b3'](out)
    out = self.features['maxpool'](out)
    out = self.features['a4'](out)
    out = self.features['b4'](out)
    out = self.features['c4'](out)
    out = self.features['d4'](out)
    out = self.features['e4'](out)
    out = self.features['maxpool'](out)
    out = self.features['a5'](out)
    out = self.features['b5'](out)
    out = self.features['avgpool'](out)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    if self.top_layer:
      out = self.top_layer(out)
    return out

  def _initialize_weights(self):
    for k in self.features:
      for i,m in enumerate(self.features[k].modules()):
        if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          for i in range(m.out_channels):
            m.weight.data[i].normal_(0, math.sqrt(2. / n))
          if m.bias is not None:
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
          m.weight.data.normal_(0, 0.01)
          m.bias.data.zero_()
    for y,m in enumerate(self.modules()):
      if isinstance(m, nn.Conv2d):
        #print(y)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        for i in range(m.out_channels):
          m.weight.data[i].normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def make_features(input_dim):
  pre_layer1 = nn.Sequential(
      OrderedDict([
        ('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3))),
        ('relu1', nn.ReLU(True)),
        ('pool1', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)),
        ])
      )
  pre_layer2 = nn.Sequential(
      OrderedDict([
        ('3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))),
        ('relu1', nn.ReLU(True)),
        ('3x3', nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1))),
        ('relu2', nn.ReLU(True)),
        ('pool2', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True))
        ])
      )

  features = {
    'pre1': pre_layer1,
    'pre2': pre_layer2,
    'a3': Inception(192,  64,  96, 128, 16, 32, 32),
    'b3': Inception(256, 128, 128, 192, 32, 96, 64),

    'maxpool': nn.MaxPool2d(3, stride=2, padding=1),

    'a4': Inception(480, 192,  96, 208, 16,  48,  64),
    'b4': Inception(512, 160, 112, 224, 24,  64,  64),
    'c4': Inception(512, 128, 128, 256, 24,  64,  64),
    'd4': Inception(512, 112, 144, 288, 32,  64,  64),
    'e4': Inception(528, 256, 160, 320, 32, 128, 128),

    'a5': Inception(832, 256, 160, 320, 32, 128, 128),
    'b5': Inception(832, 384, 192, 384, 48, 128, 128),

    'avgpool': nn.AvgPool2d(7, stride=1)
  }
  return features

def inceptionv1(sobel=False, out=1000):
  dim = 2 + int(not sobel)
  model = GoogLeNet(make_features(dim), out, sobel)
  return model
