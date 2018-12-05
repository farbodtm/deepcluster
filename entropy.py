import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class EntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(EntropyLoss, self).__init__()

    def forward(self, inputs):
        n = inputs.size(0)
        softmax = nn.Softmax(dim=1)
        inputs = softmax(inputs)
        print inputs

        # Add epsilon to avoid nan
        inputs = inputs + (10 ** (-30))

        ilog = torch.log2(inputs)

        entropy_ind = torch.sum(-(torch.sum(inputs * ilog, dim=1)))/n
        print 'Sample entropy: {}'.format(entropy_ind)

        input_sum = torch.sum(inputs, dim=0)/n
        print input_sum
        ilog2 = torch.log2(input_sum)
        entropy_total = -(torch.sum(input_sum * ilog2))
        print 'Total entropy: {}'.format(entropy_total)

        #loss = entropy_ind  + 2.5*(6.643856 - entropy_total)
        loss = entropy_ind + 0.7 * (6.643856 - entropy_total)
        return loss


def main():
    data_size = 10
    input_dim = 5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    print(EntropyLoss()(x))


if __name__ == '__main__':
    main()
    print('Running loss test...')


