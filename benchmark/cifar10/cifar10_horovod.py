'''
Cifar10 example in pytorch Horovod
'''

from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd
import os
import time
from pytorchcifargitmodels import *

model_names = {
    'resnet18'   : cifar10_resnet18(),
    'resnet34'   : cifar10_resnet34(),
    'resnet50'   : cifar10_resnet50(),
    'resnet101'  : cifar10_resnet101(),
    'resnet152'  : cifar10_resnet152(),
    'vgg11'      : cifar10_vgg11(),
    'vgg13'      : cifar10_vgg13(),
    'vgg16'      : cifar10_vgg16(),
    'vgg19'      : cifar10_vgg19(),
    'mobilenetv2': cifar10_mobilenetv2(),
}

datadir    = os.environ['DATADIR']
epochs     = int(os.environ['EPOCHS'])
batch_size = int(os.environ['BATCH_SIZE'])
log_inter  = int(os.environ['LOGINTER'])
cores_gpu  = int(os.environ['CORES_GPU'])

def train(epoch, args, model, train_sampler, train_loader, optimizer, test_sampler, test_loader):
    tick = time.time()
    steps = len(train_loader)
    data_trained = 0
    loss_sum = torch.zeros(1).cuda()
    model.train()

    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for i, (data, target) in enumerate(train_loader):
        data_trained += batch_size

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
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

    tock = time.time()

    train_loss = loss_sum.item() / data_trained
    valid_loss, valid_accuracy = test(model, test_sampler, test_loader)

    elapsed_time = tock - tick
    throughput = data_trained / elapsed_time
    print('%d/%d epoch | train loss:%.3f %.3f samples/sec | '
        'valid loss:%.3f accuracy:%.3f'
        '' % (epoch+1, epochs, train_loss, throughput,
                valid_loss, valid_accuracy))

    return throughput, elapsed_time


def test(model, test_sampler, test_loader):
    tick = time.time()
    steps = len(test_loader)
    data_tested = 0
    loss_sum = torch.zeros(1).cuda()
    accuracy_sum = torch.zeros(1).cuda()
    model.eval()

    for i, (data, target) in enumerate(test_loader):
        current_batch = data.size(0)
        data_tested += current_batch
        data, target = data.cuda(), target.cuda()
        output = model(data)

        # sum up batch loss
        loss_sum += F.cross_entropy(output, target, size_average=False).item()

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

    # Horovod: average metric values across workers.
    test_loss = metric_average(loss, 'avg_loss')
    test_accuracy = metric_average(accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTEST SET: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))

    return loss.item(), accuracy.item()


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='D-DNN cifar10 benchmark')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    # Value of args.synthetic_data may seem confusing, but those values 
    # come from bash and there 0=true and all else =false
    parser.add_argument('-s', '--synthetic_data', type=int, default=0,
                        help="Use synthetic data")
    args = parser.parse_args()

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(1)

    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(1)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(cores_gpu)


    if args.synthetic_data:
        train_dataset = \
            datasets.CIFAR10(datadir, train=True, download=False,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
        
        test_dataset = \
            datasets.CIFAR10(datadir, train=False, download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
    else:
        # Load synthetic data
        traindir = datadir + '/CIFAR10/train'
        valdir   = datadir + '/CIFAR10/val'

        train_dataset = \
            datasets.ImageFolder(traindir,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

        test_dataset = \
            datasets.ImageFolder(valdir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    # Horovod: use DistributedSampler to partition the training data.
    kwargs = {'num_workers': cores_gpu, 'pin_memory': True}
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)

    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            sampler=test_sampler, **kwargs)

    # Move model to GPU.
    print("=> creating model '{}'".format(args.arch))
    model = model_names[args.arch].cuda()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                        momentum=0.9, weight_decay=5e-4)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizedef accuracy(output, target):
    optimizer = hvd.DistributedOptimizer(optimizer,
                                        named_parameters=model.named_parameters(),
                                        op=hvd.Average)

    throughputs = []
    elapsed_times = []
    for epoch in range(0, epochs):
        throughput, elapsed_time = train(epoch, args, model, train_sampler, train_loader, optimizer, test_sampler, test_loader)
        
        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)
    
    _, valid_accuracy = test(model, test_sampler, test_loader)

    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))
