# Run
Can run 1 or multiple benchmarks at the same time. Output can be found in ~/out. It includes the installation of all software, configured for SURF's LISA cluster. All Python packages are installed through pip for Python3. For all system packages, the version has been added to the installation so it can be replicated on other systems (module load NCCL/2.5.6-CUDA-10.1.243 for example). It is created to work with SLURM. Note PyTorch, Horovod and GPipe share the same python environment in ~/.envs, while PipeDream uses a different one as the PyTorch version is different.

## Usage
    ./run.sh -h

## Environment variables
Several environment variables are set in run_template.sh, which can be read in benchmark files. These include:

* CORES_GPU: Number of CPU cores available per GPU on a node. For example if you have a 12 core CPU and 4 GPUs on a node, this value is 3. Is used to set the number of workers for dataloader objects
* EPOCHS: Number of epochs to run
* LOGINTER: Print statistics every X batches
* DATADIR: Location of the dataset
* BATCH_SIZE: Batch size
* MICROBATCHES: Number of microbatches for GPIPE.

## Advanced configurations
For more information, see the comments in the scripts. Feel free to edit these scripts. For PipeDream, a separate ssh session will be created for each node you use with PipeDream. This ssh session executed pipedream_run.sh, which will spawn a process per GPU. Alternatively you can use the scripts in the pipedream repository but those are currently not supported in this project.