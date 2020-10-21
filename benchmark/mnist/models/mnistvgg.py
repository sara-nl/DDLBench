# Credits to:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
# For providing a VGG - CIFAR-10 version which was slightly changed to fit MNIST
# 
# Changes:
# - in_channels: 3 -> 1
# - Removed the last pooling layer as 28 / 2^5 < 0 so network would crash.
# 
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

__all__ = ['MNISTVGG', 'mnist_vgg11', 'mnist_vgg13', 'mnist_vgg16', 'mnist_vgg19']

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class MNISTVGG(nn.Module):
    def __init__(self, vgg_name):
        super(MNISTVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def mnist_vgg11(pretrained=False, **kwargs):
    model = MNISTVGG('VGG11')
    if pretrained:
        print("We dont have a pretrained model for mnist")
    return model

def mnist_vgg13(pretrained=False, **kwargs):
    model = MNISTVGG('VGG13')
    if pretrained:
        print("We dont have a pretrained model for mnist")
    return model

def mnist_vgg16(pretrained=False, **kwargs):
    model = MNISTVGG('VGG16')
    if pretrained:
        print("We dont have a pretrained model for mnist")
    return model

def mnist_vgg19(pretrained=False, **kwargs):
    model = MNISTVGG('VGG19')
    if pretrained:
        print("We dont have a pretrained model for mnist")
    return model