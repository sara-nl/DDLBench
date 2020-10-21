"""A ResNet implementation but using :class:`nn.Sequential`. :func:`resnet101`
returns a :class:`nn.Sequential` instead of ``ResNet``.

This code is transformed :mod:`torchvision.models.resnet`.

"""
from collections import OrderedDict
from typing import Any, List

from torch import nn

from .block import bottleneck, basicblock
from .flatten_sequential import flatten_sequential

__all__ = ['mnist_resnet18', 'mnist_resnet34', 'mnist_resnet50', 'mnist_resnet101', 'mnist_resnet152']

def build_resnet(layers, block, num_classes=10, inplace=False):
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

        strides = [stride] + [1]*(blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(inplanes, planes, stride, None, inplace))
            inplanes = planes * expansion
        return nn.Sequential(*layers)

    # Build ResNet as a sequential model.
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),

        ('layer1', make_layer(block, 64, layers[0], inplace=inplace)),
        ('layer2', make_layer(block, 128, layers[1], stride=2, inplace=inplace)),
        ('layer3', make_layer(block, 256, layers[2], stride=2, inplace=inplace)),
        ('layer4', make_layer(block, 512, layers[3], stride=2, inplace=inplace)),

        ('avgpool', nn.AvgPool2d(4)),
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


def mnist_resnet18(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([2, 2, 2, 2], basicblock, **kwargs)

def mnist_resnet34(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 6, 3], basicblock, **kwargs)

def mnist_resnet50(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 6, 3], bottleneck, **kwargs)

def mnist_resnet101(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 23, 3], bottleneck, **kwargs)

def mnist_resnet152(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 8, 36, 3], bottleneck, **kwargs)
