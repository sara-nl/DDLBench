from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import torchvision.models as models
import horovod.torch as hvd
import os
import math
from distutils.version import LooseVersion
import time

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

batches_per_allreduce = 1

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='D-DNN imagenet benchmark')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--lr', type=float, default=0.0125, metavar='LR',
                        help='learning rate (default: 0.0125)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
    parser.add_argument('--warmup-epochs', type=float, default=1,
                    help='number of warmup epochs')
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

    cudnn.benchmark = True

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(cores_gpu)

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
    
    train_dataset = \
        datasets.ImageFolder(traindir,
            transform=transforms.Compose(train_comp))

    val_dataset = \
        datasets.ImageFolder(valdir, 
            transform=transforms.Compose(val_comp))

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    kwargs = {'num_workers': cores_gpu, 'pin_memory': True}
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    global train_loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, **kwargs)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            sampler=val_sampler, **kwargs)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = model_names[args.arch].cuda()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                        lr=(args.lr * batches_per_allreduce * hvd.size()),
                        momentum=args.momentum, weight_decay=args.wd)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, 
        named_parameters=model.named_parameters(),
        backward_passes_per_step=batches_per_allreduce,
        op=hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Run program
    throughputs = []
    elapsed_times = []
    for epoch in range(0, epochs):
        throughput, elapsed_time = train(epoch, args, model, train_sampler, train_loader, optimizer, val_sampler, val_loader)
        
        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)
    
    _, valid_accuracy = test(model, val_sampler, val_loader)

    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))


def train(epoch, args, model, train_sampler, train_loader, optimizer, test_sampler, test_loader):
    tick = time.time()
    steps = len(train_loader)
    data_trained = 0
    loss_sum = torch.zeros(1).cuda()
    model.train()

    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(epoch, idx, args, optimizer)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        data_trained += batch_size
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach() * batch_size

        if idx % log_inter == 0:
            percent = idx / steps * 100
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

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            current_batch = data.size(0)
            data_tested += current_batch
            data, target = data.cuda(), target.cuda()
            output = model(data)

            # sum up batch loss
            loss = F.cross_entropy(output, target).item()
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

    # Horovod: average metric values across workers.
    test_loss = metric_average(loss, 'avg_loss')
    test_accuracy = metric_average(accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTEST SET: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))

    return loss.item(), accuracy.item()


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, args, optimizer):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * hvd.size() * batches_per_allreduce * lr_adj


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


if __name__ == "__main__":
    main()