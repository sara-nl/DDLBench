import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import datasets, transforms
from torchsummary import summary

from cifar10.pytorchcifargitmodels import *
from mnist.models import *

resnet_names = [
    'resnet18',
    'resnet50',
    'resnet152',
]

vgg_names = [
    'vgg11',
    'vgg16',
]

mobilenet_names = [
    'mobilenet_v2', #
]

# Get summary of all models from 1 type for 1 dataset (e.g. all resnet models for imagenet)
def print_summary(models, names, netname, size):
    device = torch.device('cpu')
    print('\n' + netname + '\n')
    for m, name in zip(models, names):
        print(name)
        devm = m.to(device)
        summary(devm, size)


def mnist():
    print("\nMNIST\n")
    size = (1, 28, 28)

    resnet = [
        mnist_resnet18(),
        mnist_resnet50(),
        mnist_resnet152(),
    ]

    vgg = [
        mnist_vgg11(),
        mnist_vgg16(),
    ]

    mobilenet = [
        mnist_mobilenetv2(),
    ]

    print_summary(resnet, resnet_names, 'RESNET MODELS', size)
    print_summary(vgg, vgg_names, 'VGG MODELS', size)
    print_summary(mobilenet, mobilenet_names, 'MOBILENET_V2 MODELS', size)


def cifar10():
    print("\nCIFAR-10\n")
    size = (3, 32, 32)

    resnet = [
        cifar10_resnet18(),
        cifar10_resnet50(),
        cifar10_resnet152(),
    ]

    vgg = [
        cifar10_vgg11(),
        cifar10_vgg16(),
    ]

    mobilenet = [
        cifar10_mobilenetv2(),
    ]

    print_summary(resnet, resnet_names, 'RESNET MODELS', size)
    print_summary(vgg, vgg_names, 'VGG MODELS', size)
    print_summary(mobilenet, mobilenet_names, 'MOBILENET_V2 MODELS', size)


def imagenet1000():
    print("\nIMAGENET-1000\n")
    size = (3, 224, 224)

    resnet = [
        models.resnet18(),
        models.resnet50(),
        models.resnet152(),
    ]

    vgg = [
        models.vgg11(),
        models.vgg16(),
    ]

    mobilenet = [
        models.mobilenet_v2(),
    ]

    print_summary(resnet, resnet_names, 'RESNET MODELS', size)
    print_summary(vgg, vgg_names, 'VGG MODELS', size)
    print_summary(mobilenet, mobilenet_names, 'MOBILENET_V2 MODELS', size)


if __name__ == '__main__':
    mnist()
    cifar10()
    imagenet1000()
