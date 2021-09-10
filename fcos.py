import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def conv3x3(in_planes, out_planes, stride=1):
    """returns a 3x3 convolution with padding=1"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()

class FCOS(nn.Module):
    def __init__(self):
        super(FCOS, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    def train(epochs):
        return None


    def test():
        return None
