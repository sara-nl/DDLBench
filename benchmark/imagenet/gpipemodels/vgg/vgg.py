'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ['inet_vgg11', 'inet_vgg13', 'inet_vgg16', 'inet_vgg19']

from .flatten_sequential import flatten_sequential

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def build_vgg(vgg_name):
    def make_layers(cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=False)] # Changed to false for inplace error
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((7, 7))] # Was: nn.AvgPool2d(kernel_size=1, stride=1)

        return nn.Sequential(*layers)

    classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=False), # Changed to false for inplace error (gpipe autograd)
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=False), # Changed to false for inplace error
        nn.Dropout(),
        nn.Linear(4096, 1000),
    )

    # Build a sequential model
    model = nn.Sequential(OrderedDict([
        ('features', make_layers(cfg[vgg_name])),
        ('flat', nn.Flatten()),
        ('classifier', classifier),
    ]))

    # Flatten nested sequentials.
    model = flatten_sequential(model)
    return model


def inet_vgg11(pretrained=False, **kwargs):
    model = build_vgg('VGG11')
    if pretrained:
        print("We dont have a pretrained model for cifar10")
    return model

def inet_vgg13(pretrained=False, **kwargs):
    model = build_vgg('VGG13')
    if pretrained:
        print("We dont have a pretrained model for cifar10")
    return model

def inet_vgg16(pretrained=False, **kwargs):
    model = build_vgg('VGG16')
    if pretrained:
        print("We dont have a pretrained model for cifar10")
    return model

def inet_vgg19(pretrained=False, **kwargs):
    model = build_vgg('VGG19')
    if pretrained:
        print("We dont have a pretrained model for cifar10")
    return model