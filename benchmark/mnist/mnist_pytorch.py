'''
Mnist example in pytorch
https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py
'''

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch.optim as optim
from torchvision import datasets, transforms
import time
from models import *

model_names = {
    'resnet18'   : mnist_resnet18(),
    'resnet34'   : mnist_resnet34(),
    'resnet50'   : mnist_resnet50(),
    'resnet101'  : mnist_resnet101(),
    'resnet152'  : mnist_resnet152(),
    'vgg11'      : mnist_vgg11(),
    'vgg13'      : mnist_vgg13(),
    'vgg16'      : mnist_vgg16(),
    'vgg19'      : mnist_vgg19(),
    'mobilenetv2': mnist_mobilenetv2(),
}

datadir    = os.environ['DATADIR']
epochs     = int(os.environ['EPOCHS'])
batch_size = int(os.environ['BATCH_SIZE'])
log_inter  = int(os.environ['LOGINTER'])
cores_gpu  = int(os.environ['CORES_GPU'])


def train(args, model, device, train_loader, test_loader):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    throughputs = []
    elapsed_times = []
    for epoch in range(0, epochs):
        throughput, elapsed_time = train_epoch(epoch, args, model, device, train_loader, optimizer, test_loader)

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)
    
    return throughputs, elapsed_times


def train_epoch(epoch, args, model, device, train_loader, optimizer, test_loader):
    torch.cuda.synchronize(device)
    tick = time.time()
    steps = len(train_loader)
    data_trained = 0
    loss_sum = torch.zeros(1, device=device)
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data_trained += batch_size

        output = model(data.to(device))
        loss = F.cross_entropy(output, target.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach() * batch_size

        if i % log_inter == 0:
            percent = i / steps * 100
            throughput = data_trained / (time.time()-tick)

            dev = torch.cuda.current_device()
            stats = torch.cuda.memory_stats(device=dev)
            max_mem = torch.cuda.get_device_properties(dev).total_memory
            print('train | %d/%d epoch (%d%%) | %.3f samples/sec (estimated) | mem (GB): %.3f (%.3f) / %.3f'
                '' % (epoch+1, epochs, percent, throughput, 
                      stats["allocated_bytes.all.peak"] / 10**9,
                      stats["reserved_bytes.all.peak"] / 10**9,
                      float(max_mem) / 10**9))
    
    torch.cuda.synchronize(device)
    tock = time.time()

    train_loss = loss_sum.item() / data_trained
    valid_loss, valid_accuracy = test_epoch(model, device, test_loader)
    torch.cuda.synchronize(device)

    elapsed_time = tock - tick
    throughput = data_trained / elapsed_time
    print('%d/%d epoch | train loss:%.3f %.3f samples/sec | '
        'valid loss:%.3f accuracy:%.3f'
        '' % (epoch+1, epochs, train_loss, throughput,
                valid_loss, valid_accuracy))

    return throughput, elapsed_time


def test_epoch(model, device, test_loader):
    tick = time.time()
    steps = len(test_loader)
    data_tested = 0
    loss_sum = torch.zeros(1, device=device)
    accuracy_sum = torch.zeros(1, device=device)
    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            current_batch = data.size(0)
            data_tested += current_batch

            target = target.to(device)
            output = model(data.to(device))
            loss = F.cross_entropy(output, target)
            loss_sum += loss * current_batch

            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum()
            accuracy_sum += correct

            if i % log_inter == 0:
                percent = i / steps * 100
                throughput = data_tested / (time.time() - tick)
                print('valid | %d%% | %.3f samples/sec (estimated)'
                    '' % (percent, throughput))

    loss = loss_sum / data_tested
    accuracy = accuracy_sum / data_tested

    return loss.item(), accuracy.item()


def grayloader(path):
    # TorchVision imageloader does not support grayscale out of the box
    from PIL import Image
    import os.path
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='D-DNN mnist benchmark')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')    
    # Value of args.synthetic_data may seem confusing, but those values 
    # come from bash and there 0=true and all else =false
    parser.add_argument('-s', '--synthetic_data', type=int, default=0,
                        help="Use synthetic data")
    args = parser.parse_args()

    device = torch.device("cuda")
    dataloader_kwargs = {'pin_memory': True}

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    print("=> creating model '{}'".format(args.arch))
    model = model_names[args.arch].to(device)

    if args.synthetic_data:
        # Load normal data
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(datadir, train=True, download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True, num_workers=cores_gpu,
            **dataloader_kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(datadir, train=False, download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True, num_workers=cores_gpu,
            **dataloader_kwargs)
    else:
        # Load synthetic data
        traindir = datadir + '/MNIST/train'
        valdir   = datadir + '/MNIST/val'

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]),
                loader=grayloader,
            ),
            batch_size=batch_size, shuffle=True,
            num_workers=cores_gpu, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir, 
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]),
                loader=grayloader,
            ),
            batch_size=batch_size, shuffle=True,
            num_workers=cores_gpu, pin_memory=True)

    # Run the model
    throughputs, elapsed_times = train(args, model, device, train_loader, test_loader)
    _, valid_accuracy = test_epoch(model, device, test_loader)

    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))


if __name__ == '__main__':
    main()
