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
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

model_names = {
    'resnet18'   : models.resnet18(),
    'resnet34'   : models.resnet34(),
    'resnet50'   : models.resnet50(),
    'resnet101'  : models.resnet101(),
    'resnet152'  : models.resnet152(),
    'vgg11'      : models.vgg11(),
    'vgg13'      : models.vgg13(),
    'vgg16'      : models.vgg16(),
    'vgg19'      : models.vgg19(),
    'mobilenetv2': models.mobilenet_v2(),
}

datadir    = os.environ['DATADIR']
epochs     = int(os.environ['EPOCHS'])
batch_size = int(os.environ['BATCH_SIZE'])
log_inter  = int(os.environ['LOGINTER'])
cores_gpu  = int(os.environ['CORES_GPU'])

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

def main():
    args = parser.parse_args()
    torch.manual_seed(1)        # Not setting cudnn deterministic because that makes it slower
    torch.cuda.manual_seed(1)   # And I dont really care about accuracy

    # create model
    print("=> creating model '{}'".format(args.arch))
    device = torch.device("cuda")
    model = model_names[args.arch].cuda()

    cudnn.benchmark = True
    device = torch.device("cuda")

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

    # Run the model
    throughputs, elapsed_times = train(args, model, device, train_loader, val_loader)
    _, valid_accuracy = test_epoch(model, device, val_loader)

    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))


def train(args, model, device, train_loader, test_loader):
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    throughputs = []
    elapsed_times = []
    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch, args)
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
