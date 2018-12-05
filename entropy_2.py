import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class EntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(EntropyLoss, self).__init__()

    def forward(self, inputs, total=False):
        n = inputs.size(0)
        softmax = nn.Softmax(dim=1)
        inputs = softmax(inputs)

        # Add epsilon to avoid nan
        inputs = inputs + (10 ** (-30))

        #if not total:
        #    ilog = torch.log2(inputs)

        #    entropy_ind = torch.sum(-(torch.sum(inputs * ilog, dim=1)))/n
        #    print 'Sample entropy: {}'.format(entropy_ind)
        #    loss = entropy_ind
        #else:
        #    input_sum = torch.sum(inputs, dim=0)/n
        #    print input_sum
        #    ilog2 = torch.log2(input_sum)
        #    entropy_total = -(torch.sum(input_sum * ilog2))
        #    print 'Total entropy: {}'.format(entropy_total)

        #    loss = 100*(6.643856 - entropy_total)

        ilog = torch.log2(inputs)

        entropy_ind = torch.sum(-(torch.sum(inputs * ilog, dim=1)))/n
        print 'Sample entropy: {}'.format(entropy_ind)
        loss = entropy_ind
        input_sum = torch.sum(inputs, dim=0)/n
        print input_sum
        ilog2 = torch.log2(input_sum)
        entropy_total = -(torch.sum(input_sum * ilog2))
        print 'Total entropy: {}'.format(entropy_total)

        loss = loss + 60 * ((6.643856 - entropy_total))
        return loss


def main():
    data_size = 10
    input_dim = 5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    print(EntropyLoss()(x, total=True))


if __name__ == '__main__':
    main()
    print('Running loss test...')


