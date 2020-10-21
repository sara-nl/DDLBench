"""A ResNet implementation but using :class:`nn.Sequential`. :func:`resnet101`
returns a :class:`nn.Sequential` instead of ``ResNet``.

This code is transformed :mod:`torchvision.models.resnet`.

"""
from collections import OrderedDict
from typing import Any, List

from torch import nn

from .block import bottleneck, basicblock
from .flatten_sequential import flatten_sequential

__all__ = ['inet_resnet18', 'inet_resnet34', 'inet_resnet50', 'inet_resnet101', 'inet_resnet152']

def build_resnet(layers, block, num_classes=1000, inplace=False):
    """Builds a ResNet as a simple sequential model.

    Note:
        The implementation is copied from :mod:`torchvision.models.resnet`.

    """
    inplanes = 64
    if block.__name__ == 'basicblock':
        expansion = 1
    elif block.__name__ == 'bottleneck':
        expansion = 4
    else:
        raise Exception('Invalid block')

    def make_layer(block, planes, blocks, stride=1, inplace=False):
        nonlocal inplanes
        nonlocal expansion

        downsample = None
        if stride != 1 or inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, inplace))
        inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, inplace=inplace))

        return nn.Sequential(*layers)

    # Build ResNet as a sequential model.
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('layer1', make_layer(block, 64, layers[0], inplace=inplace)),
        ('layer2', make_layer(block, 128, layers[1], stride=2, inplace=inplace)),
        ('layer3', make_layer(block, 256, layers[2], stride=2, inplace=inplace)),
        ('layer4', make_layer(block, 512, layers[3], stride=2, inplace=inplace)),

        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flat', nn.Flatten()),
        ('fc', nn.Linear(512 * expansion, num_classes)),
    ]))

    # Flatten nested sequentials.
    model = flatten_sequential(model)

    # Initialize weights for Conv2d and BatchNorm2d layers.
    # Stolen from torchvision-0.4.0.
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            return

        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            return

    model.apply(init_weight)

    return model


def inet_resnet18(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([2, 2, 2, 2], basicblock, **kwargs)

def inet_resnet34(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 6, 3], basicblock, **kwargs)

def inet_resnet50(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 6, 3], bottleneck, **kwargs)

def inet_resnet101(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 23, 3], bottleneck, **kwargs)

def inet_resnet152(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 8, 36, 3], bottleneck, **kwargs)
