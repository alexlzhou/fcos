import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, self.expansion * channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

    def _make_layer(self, block, channels, num_blocks, stride):

    def _upsample_add(self, x, y):

    def forward(self, x):