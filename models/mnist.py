import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [ 'Mnist', 'mnist']
 

class Mnist(nn.Module):
    def __init__(self, num_classes, sobel):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.classifier = nn.Sequential(
                            nn.Linear(320, 50),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),)

        self.top_layer = nn.Linear(50, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.classifier(x)
        x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
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


def mnist(sobel=False, bn=True, out=10):
    model = Mnist(out, sobel)
    return model
