# Pipedream for Distribued deep learning benchmark suite
The pipedream source code was edited to fit this project, see /pipedream-fork. It is based on:

    https://github.com/msr-fiddle/pipedream/tree/f50827f2e28cbdbd82a4ea686c0498272b1460d6

If you are not interested in how PipeDream was changed, skip this README. Support for MNIST, CIFAR-10 and highres has been added, as well as for more models. This, along with several bug and quality-of-life fixes has resulted in multiple source code changes.

To find all source code changes:

    cd pipedream-fork
    grep -r "SOURCE"

## Installation
Pipedream needs a particular version of PyTorch (a commit just before v1.1.0) and matching packages. You can either use the nvidia-docker that comes with PyTorch or use the custom PyTorch wheel in:

    ./torch_pipedream/pytorch/dist/torch-1.1.0a0+828a6a3-cp37-cp37m-linux_x86_64.whl

Which is automatically used in the run scripts. Building PyTorch yourself for Pipedream can be very difficult as PipeDream includes a patch (diff file) which only works for a specific PyTorch version used in the Nvidia-docker container which is included in the PipeDream repository. 

## Change 1: Importing data sets and networks 
When adding new benchmarks / data sets to the benchmark suite, logic has to be added to pipedream so it can import these. The most important parts to edit are:

    profiler/image_classification/main.py
    profiler/image_classification/models/
    profiler/image_classification/models/
    runtime/image_classification/main_with_runtime.py

PipeDream executes scripts in this order: First the profiler creates an intermediate representation (a graph) of PyTorch's neural network object. Secondly the optimizer interprets this graph in a distributed setting and creates an optimized model (the same PyTorch object format which you gave to the PipeDream at the start). Finally the runtime executes the optimized model. The optimizer does not need the actual data as it uses the intermediate representation so only the profiler and runtime need to be changed when adding data sets / networks. In both python files you need to add dataloader logic / network importing logic. Furthermore the network must be present in the /models folder, which the run scripts of this repository fix by creating symlinks to the /benchmark folder.

## Change 2: Modifying the intermediate representation
As PipeDream parses a PyTorch model to a custom intermediate presentation, all layers of your model should be supported by PipeDream. The following rules are very important if you do not want to change major parts of the source code (mainly the profiler and optimizer):

* Do not use PyTorch functional library (mostly imported as F). Instead use nn. type layers
* Try to use flatten instead of view to resize layers if possible

If you still get errors, you probably have to edit the intermediate representation. The following files may be changed (and have already been changed):

    profiler/torchmodules/torchgraph/graph_creator.py
    optimizer/convert_graph_to_model.py

## Fixes
The following files have been changed to steamline some parts of PipeDream:

    runtime/image_classification/main_with_runtime.py
    runtime/runtime.py
    runtime/communication.py

The first one fixes the problem that when using the --stage_to_num_ranks argument with data parallelism, for example 0:4 (execute stage 0 with 4 gpu's). This should lead to data parallelism but PipeDream only uses full data parallelism when the user does not give this argument, which is non intuitive.

The second file has been changed to fix the problem of the number of batches in a data set not being divisible by the number of GPUs. Example: PipeDream uses a 1-3 config (split model in 2 parts, first part to be executed by 1 GPU, the second part by 3 GPUs in data parallel), dataset contains 1000 batches. GPU 0 uses 1000 / 1 = 1000 batches, so the 3 GPUs in phase two want to use 1000 / 3 = 333.3 batches per GPU. However this is not perfectly divisible so PipeDream crashes. This is solved by using the gcd() (greatest common denominator), so PipEDream finds a perfect number of batches so for all phases with any number of GPUs it is perfectly divisible.

The third file has been changed in setup_messaging_schedule, otherwise all configs like 1-3 where a previous phase uses less GPUs than the current phase would result in a deadlock, and eventually a timeout.