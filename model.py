import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models.vgg import VGG, cfg, make_layers, vgg16

from config import Config

device = torch.device(Config.DEVICE)

class Scale(nn.Module):

    def __init__(self, initialized_factor):
        super().__init__()
        self.factor = torch.tensor(
            initialized_factor, requires_grad=True
        ).float().to(device)
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return x / norm * self.factor


class Net(VGG):

    def __init__(self):
        super().__init__(make_layers(cfg['D']))

        self.pool5 = self.features[30]
        self.conv_fc6 = self._conv_block(512, 1024)
        self.conv_fc7 = self._conv_block(1024, 1024, kernel=1)

        self.conv6_1 = self._conv_block(1024, 256, kernel=1)
        self.conv6_2 = self._conv_block(256, 512, stride=2)

        self.conv7_1 = self._conv_block(512, 128, kernel=1)
        self.conv7_2 = self._conv_block(128, 256, stride=2)

        self.norm3_3 = Scale(10)
        self.norm4_3 = Scale(8)
        self.norm5_3 = Scale(5)

        self.predict3_3 = nn.Conv2d(256, 6, kernel_size=3, padding=1)
        self.predict4_3 = nn.Conv2d(512, 6, kernel_size=3, padding=1)
        self.predict5_3 = nn.Conv2d(512, 6, kernel_size=3, padding=1)
        self.predict_fc7 = nn.Conv2d(1024, 6, kernel_size=3, padding=1)
        self.predict6_2 = nn.Conv2d(512, 6, kernel_size=3, padding=1)
        self.predict7_2 = nn.Conv2d(256, 6, kernel_size=3, padding=1)

    def stride_forward(self, x, start, end):
        for layer in self.features[start:end]:
            x = layer(x)
        return x

    def forward(self, x):
        f1 = self.stride_forward(x, 0, 16)
        f2 = self.stride_forward(f1, 16, 23)
        f3 = self.stride_forward(f2, 23, 30)

        x = self.pool5(f3)
        x = self.conv_fc6(x)
        f4 = self.conv_fc7(x)

        x = self.conv6_1(f4)
        f5 = self.conv6_2(x)

        x = self.conv7_1(f5)
        f6 = self.conv7_2(x)

        return [
            # F.relu(self.predict3_3(f1)), # ignore the first layer
            self.predict4_3(self.norm4_3(f2)),
            self.predict5_3(self.norm5_3(f3)),
            self.predict_fc7(f4),
            self.predict6_2(f5),
            self.predict7_2(f6)
        ]

    def _conv_block(self, in_channel, out_channel, kernel=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                      padding=kernel // 2, stride=stride),
            nn.ReLU(inplace=True)
        )
