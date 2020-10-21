import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time

from typing import cast
import time
from collections import OrderedDict

from gpipemodels.resnet import *
from gpipemodels.vgg import *
from gpipemodels.mobilenetv2 import *

model_names = {
    'resnet18'   : inet_resnet18(),
    'resnet50'   : inet_resnet50(),
    'resnet152'  : inet_resnet152(),
    'vgg11'      : inet_vgg11(),
    'vgg16'      : inet_vgg16(),
    'mobilenetv2': inet_mobilenetv2(),
}

datadir         = os.environ['DATADIR']
epochs          = int(os.environ['EPOCHS'])
microbatch_size = int(os.environ['BATCH_SIZE'])
log_inter       = 1
cores_gpu       = int(os.environ['CORES_GPU'])
microbatches    = int(os.environ['MICROBATCHES'])
batch_size      = microbatch_size * microbatches

def main():
    parser = argparse.ArgumentParser(description='D-DNN imagenet benchmark')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # Value of args.synthetic_data may seem confusing, but those values 
    # come from bash and there 0=true and all else =false
    parser.add_argument('-s', '--synthetic_data', type=int, default=0,
                        help="Use synthetic data")
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    cudnn.benchmark = True

    #---------------------------------------------------------------------------------
    # Move model to GPU.
    print("=> creating model '{}'".format(args.arch))
    model = model_names[args.arch].cuda()

    partitions = torch.cuda.device_count()
    if args.synthetic_data == -1:
        sample = torch.empty(batch_size, 3, 512, 512)
    else:
        sample = torch.empty(batch_size, 3, 224, 224)
    balance = balance_by_time(partitions, model, sample)
    model = GPipe(model, balance, chunks=microbatches)
    
    #---------------------------------------------------------------------------------
    devices = list(model.devices)
    in_device  = devices[0]
    out_device = devices[-1]
    torch.cuda.set_device(in_device)

    throughputs = []
    elapsed_times = []
    #---------------------------------------------------------------------------------

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    #---------------------------------------------------------------------------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_comp = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]
    val_comp   = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]

    if args.synthetic_data == -1:
        # Load highres data
        traindir = datadir + '/HIGHRES/train'
        valdir   = datadir + '/HIGHRES/val'
        train_comp = [transforms.ToTensor(), normalize]
        val_comp   = [transforms.ToTensor(), normalize]
    elif args.synthetic_data:
        # Load normal data
        traindir = datadir + '/train'
        valdir   = datadir + '/val'
    else:
        # Load synthetic data
        traindir = datadir + '/IMAGENET/train'
        valdir   = datadir + '/IMAGENET/val'
    
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose(train_comp)), 
        batch_size=batch_size, shuffle=True,
        num_workers=cores_gpu, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir, 
            transforms.Compose(val_comp)),
        batch_size=batch_size, shuffle=True,
        num_workers=cores_gpu, pin_memory=True)
    #---------------------------------------------------------------------------------

    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(train_loader, val_loader, model, optimizer, epoch, args, in_device, out_device)

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

    _, valid_accuracy = evaluate(val_loader, model, args, in_device, out_device)

    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))


def run_epoch(train_loader, test_loader, model, optimizer, epoch, args, in_device, out_device):
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
        loss = F.cross_entropy(output, target)

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
    valid_loss, valid_accuracy = evaluate(test_loader, model, args, in_device, out_device)
    torch.cuda.synchronize(in_device)

    elapsed_time = tock - tick
    throughput = data_trained / elapsed_time
    print('%d/%d epoch | train loss:%.3f %.3f samples/sec | '
        'valid loss:%.3f accuracy:%.3f'
        '' % (epoch+1, epochs, train_loss, throughput,
                valid_loss, valid_accuracy))

    return throughput, elapsed_time


def evaluate(test_loader, model, args, in_device, out_device):
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


if __name__ == '__main__':
    main()