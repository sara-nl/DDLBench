# Generate synthetic data for mnist, cifar10 and / or imagenet
# Creates 1000 random tensors at a time, writes them to .JPEG
# using PIL. Supports either (grayscale, 1 channel, L) or (color, 3 channels, RGB)

import torch
import sys
import os
import datetime
import multiprocessing
import numpy as np

from torchvision.utils import save_image
from multiprocessing import Pool
from functools import partial
from torchvision import transforms

datadir = os.environ['DATADIR']

# Writes all images for 1 class to a directory as .JPEG
# Is used as multithreaded function: Each thread = 1 class
def generate(length, dimension, channels, subdir, classes):
    for classnr in classes:
        os.mkdir(datadir + subdir + "class_" + str(classnr))
        classpath = datadir + subdir + "class_" + str(classnr) + "/"

        if channels == 1:
            mode = "L"
        else:
            mode = "RGB"

        # Write in batches of 1000
        togo = length
        img_nr = 0
        while togo > 0:
            bsize = 1000
            if togo < 1000:
                bsize = togo

            images = [torch.rand(channels, dimension, dimension) for _ in range(bsize)]
            for img in images:
                im = transforms.ToPILImage(mode=mode)(img)
                im.save(classpath + "img_" + str(img_nr) + '.JPEG')
                img_nr += 1

            togo -= 1000


# Blueprint for each dataset
def dataset(trainlength, testlength, dimension, channels, nrclasses, setname):
    os.mkdir(datadir + "/" + setname)
    os.mkdir(datadir + "/" + setname + "/train")
    os.mkdir(datadir + "/" + setname + "/val")

    trainlength = int(trainlength / nrclasses)
    testlength  = int(testlength / nrclasses)

    # Spawn nr threads = nr. of cores, divide classes over cores
    nrcores = multiprocessing.cpu_count()
    if nrcores > nrclasses:
        nrcores = nrclasses

    print("Use " + str(nrcores) + " cores with " + str(nrclasses) + " classes in total", flush=True)
    split = np.array_split(range(nrclasses), nrcores)

    with Pool(nrcores) as p:
        func = partial(generate, trainlength, dimension, channels, "/" + setname + "/train/")
        p.map(func, split)

    with Pool(nrcores) as p:
        func = partial(generate, testlength, dimension, channels, "/" + setname + "/val/")
        p.map(func, split)


# Can generate multiple datasets
def main(datasets):
    if "mnist" in datasets:
        start = datetime.datetime.now()
        print("Start mnist " + str(start), flush=True)
        dataset(60000, 10000, 28, 1, 10, "MNIST")
        end   = datetime.datetime.now()
        delta = (end - start).total_seconds()
        print('took (sec): ' + str(delta) + ' | imgs / sec: ' + 
                str(float(60000 + 10000) / delta), flush=True)
    if "cifar10" in datasets:
        start = datetime.datetime.now()
        print("Start cifar10 " + str(start), flush=True)
        dataset(50000, 10000, 32, 3, 10, "CIFAR10")
        end   = datetime.datetime.now()
        delta = (end - start).total_seconds()
        print('took (sec): ' + str(delta) + ' | imgs / sec: ' + 
                str(float(50000 + 10000) / delta), flush=True)
    if "imagenet" in datasets:
        start = datetime.datetime.now()
        print("Start imagenet " + str(start), flush=True)
        dataset(1281167, 50000, 224, 3, 1000, "IMAGENET")
        end   = datetime.datetime.now()
        delta = (end - start).total_seconds()
        print('took (sec): ' + str(delta) + ' | imgs / sec: ' + 
                str(float(1281167 + 50000) / delta), flush=True)
    if "highres" in datasets:
        start = datetime.datetime.now()
        print("Start highres " + str(start), flush=True)
        dataset(50000, 10000, 512, 3, 1000, "HIGHRES")
        end   = datetime.datetime.now()
        delta = (end - start).total_seconds()
        print('took (sec): ' + str(delta) + ' | imgs / sec: ' + 
                str(float(50000 + 10000) / delta), flush=True)



if __name__ == '__main__':
    args = []
    for arg in sys.argv[1:]:
        args.append(str(arg))

    main(args)
