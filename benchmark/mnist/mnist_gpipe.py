'''
Mnist example in pytorch
https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py
'''

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch.optim as optim
from torchvision import datasets, transforms

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time

from typing import cast
import time
from collections import OrderedDict

from gpipemodels.resnet import *
from gpipemodels.vgg import *
from gpipemodels.mobilenetv2 import *

model_names = {
    'resnet18'   : mnist_resnet18(),
    'resnet50'   : mnist_resnet50(),
    'resnet152'  : mnist_resnet152(),
    'vgg11'      : mnist_vgg11(),
    'vgg16'      : mnist_vgg16(),
    'mobilenetv2': mnist_mobilenetv2(),
}

datadir         = os.environ['DATADIR']
epochs          = int(os.environ['EPOCHS'])
microbatch_size = int(os.environ['BATCH_SIZE'])
log_inter       = 1
cores_gpu       = int(os.environ['CORES_GPU'])
microbatches    = int(os.environ['MICROBATCHES'])
batch_size      = microbatch_size * microbatches

def evaluate(test_loader, in_device, out_device, model):
    tick = time.time()
    steps = len(test_loader)
    data_tested = 0
    loss_sum = torch.zeros(1, device=out_device)
    accuracy_sum = torch.zeros(1, device=out_device)
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            current_batch = input.size(0)
            data_tested += current_batch
            input = input.to(device=in_device)
            target = target.to(device=out_device)

            output = model(input)

            loss = F.nll_loss(output, target)
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


def run_epoch(args, model, in_device, out_device, train_loader, test_loader, epoch, optimizer):
    torch.cuda.synchronize(in_device)
    tick = time.time()

    steps = len(train_loader)
    data_trained = 0
    loss_sum = torch.zeros(1, device=out_device)
    model.train()
    for i, (input, target) in enumerate(train_loader):
        data_trained += batch_size
        input = input.to(device=in_device, non_blocking=True)
        target = target.to(device=out_device, non_blocking=True)

        output = model(input)
        loss = F.nll_loss(output, target)

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

    torch.cuda.synchronize(in_device)
    tock = time.time()

    train_loss = loss_sum.item() / data_trained
    valid_loss, valid_accuracy = evaluate(test_loader, in_device, out_device, model)
    torch.cuda.synchronize(in_device)

    elapsed_time = tock - tick
    throughput = data_trained / elapsed_time
    print('%d/%d epoch | train loss:%.3f %.3f samples/sec | '
        'valid loss:%.3f accuracy:%.3f'
        '' % (epoch+1, epochs, train_loss, throughput,
                valid_loss, valid_accuracy))

    return throughput, elapsed_time


def grayloader(path):
    # TorchVision imageloader does not support grayscale out of the box
    from PIL import Image
    import os.path
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


if __name__ == '__main__':
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

    #---------------------------------------------------------------------------------
    # Move model to GPU.
    print("=> creating model '{}'".format(args.arch))
    model = model_names[args.arch].cuda()

    partitions = torch.cuda.device_count()
    sample = torch.empty(batch_size, 1, 28, 28)
    balance = balance_by_time(partitions, model, sample)

    model = GPipe(model, balance, chunks=microbatches)
    
    #---------------------------------------------------------------------------------
    devices = list(model.devices)
    in_device  = devices[0]
    out_device = devices[-1]
    torch.cuda.set_device(in_device)

    throughputs = []
    elapsed_times = []
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(args, model, in_device, out_device, train_loader, test_loader, epoch, optimizer)

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

    _, valid_accuracy = evaluate(test_loader, in_device, out_device, model)

    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))
