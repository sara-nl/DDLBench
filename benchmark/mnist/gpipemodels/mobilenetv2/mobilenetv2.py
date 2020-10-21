'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
from collections import OrderedDict
from .flatten_sequential import flatten_sequential
from torchgpipe.skip import Namespace, pop, skippable, stash

__all__ = ['mnist_mobilenetv2']


@skippable(stash=['identity'])
class Identity(nn.Module):
    def forward(self, tensor):
        yield stash('identity', tensor)
        return tensor


@skippable(pop=['identity'])
class Shortcut(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, input):
        identity = yield pop('identity')
        out = input
        if self.stride == 1:
            out = input + self.shortcut(identity)
        return out


def block(in_planes, out_planes, expansion, stride):
    planes = expansion * in_planes
    layers = OrderedDict()

    ns = Namespace()
    layers['identity'] = Identity().isolate(ns)  # type: ignore

    layers['conv1'] = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
    layers['bn1'] = nn.BatchNorm2d(planes)
    layers['relu1'] = nn.ReLU(inplace=False)

    layers['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
    layers['bn2'] = nn.BatchNorm2d(planes)
    layers['relu2'] = nn.ReLU(inplace=False)

    layers['conv3'] = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
    layers['bn3'] = nn.BatchNorm2d(out_planes)

    layers['shortcut'] = Shortcut(in_planes, out_planes, stride).isolate(ns)

    return nn.Sequential(layers)


def build_mobilenetv2():
    def make_layers(in_planes, cfg):
        layers = []
        for expansion, out_planes, num_blocks, stride in cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for MNIST
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    # NOTE: change conv1 stride 2 -> 1 for MNIST
    conv1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=False),
    )

    conv2 = nn.Sequential(
        nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(1280),
        nn.ReLU(inplace=False),
    )

    model = nn.Sequential(OrderedDict([
        ('conv1', conv1),
        ('layer', make_layers(32, cfg)),
        ('conv2', conv2),
        ('avgpool', nn.AvgPool2d(4)),   # 7 -> 4
        ('flat', nn.Flatten()),
        ('linear', nn.Linear(1280, 10)),
    ]))

    # Flatten nested sequentials.
    model = flatten_sequential(model)
    return model


def mnist_mobilenetv2(pretrained=False, **kwargs):
    model = build_mobilenetv2()
    if pretrained:
        print("We dont have a pretrained model")
    return model
