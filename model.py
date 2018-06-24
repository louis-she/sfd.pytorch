import torch
from torch import nn


class Scale(nn.Module):

    def __init__(self, initialized_factor):
        super().__init__()
        self.factor = torch.tensor(initialized_factor, requires_grad=True)
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return x / norm * self.factor


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1_1 = self._conv_block(3, 64)
        self.conv1_2 = self._conv_block(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = self._conv_block(64, 128)
        self.conv2_2 = self._conv_block(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = self._conv_block(128, 256)
        self.conv3_2 = self._conv_block(256, 256)
        self.conv3_3 = self._conv_block(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = self._conv_block(256, 512)
        self.conv4_2 = self._conv_block(512, 512)
        self.conv4_3 = self._conv_block(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = self._conv_block(512, 512)
        self.conv5_2 = self._conv_block(512, 512)
        self.conv5_3 = self._conv_block(512, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_fc6 = self._conv_block(512, 1024)
        self.conv_fc7 = self._conv_block(1024, 1024, kernel=1)

        self.conv6_1 = self._conv_block(1024, 256, kernel=1)
        self.conv6_2 = self._conv_block(256, 512, stride=2)

        self.conv7_1 = self._conv_block(512, 128, kernel=1)
        self.conv7_2 = self._conv_block(128, 256, stride=2)

        self.norm3_3 = Scale(10)
        self.norm4_3 = Scale(8)
        self.norm5_3 = Scale(5)

        self.predict3_3 = nn.Conv2d(256, 6, kernel_size=3) #TODO: maxout?

        self.predict4_3 = nn.Conv2d(512, 6, kernel_size=3)
        self.predict5_3 = nn.Conv2d(512, 6, kernel_size=3)
        self.predict_fc7 = nn.Conv2d(1024, 6, kernel_size=3)
        self.predict6_2 = nn.Conv2d(512, 6, kernel_size=3)
        self.predict7_2 = nn.Conv2d(256, 6, kernel_size=3)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        f1 = self.conv3_2(x)
        x = self.conv3_3(f1)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        f2 = self.conv4_3(x)
        x = self.pool4(f2)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        f3 = self.conv5_3(x)
        x = self.pool(f3)

        x = self.conv_fc6(x)
        f4 = self.conv_fc7(x)

        x = self.conv6_1(f4)
        f5 = self.conv6_2(x)

        x = self.conv7_1(f5)
        f6 = self.conv7_1(x)

        return (
            self.predict3_3(f1),
            self.predict4_3(f2),
            self.predict5_3(f3),
            self.predict_fc7(f4),
            self.predict6_2(f5),
            self.predict7_2(f6)
        )

    def _conv_block(self, in_channel, out_channel, kernel=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                      padding=kernel % 2, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )