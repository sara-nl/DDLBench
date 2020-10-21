#!/bin/bash
#SBATCH -J benchmark_summary
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH -o model_summary.out
#SBATCH -e model_summary.out

cd ~

# Load modules and libs
mkdir -p .envs

module purge
module use ~/environment-modules-lisa
module load 2020
module load Python/3.7.4-GCCcore-8.3.0

module unload 2019
module unload GCC
module unload GCCcore
module unload binutils
module unload zlib
module unload compilerwrappers

module load OpenMPI/3.1.4-GCC-8.3.0
module load libyaml/0.2.2-GCCcore-8.3.0

export MPICC=mpicc
export MPICXX=mpicxx

python3 -m venv ~/.envs/env_recdistr
source ~/.envs/env_recdistr/bin/activate

pip3 install pyyaml

# We want cuda 10.1 torch / torchvision
pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html


#--------------------------------------------------------------
# Installs torchsummary and some more python packages that may be needed
cd ~/dnn-benchmark-suite/run
pip3 install -r requirements.txt
cd ~

#--------------------------------------------------------------
module list
pip3 freeze

cd ~/dnn-benchmark-suite/benchmark
python3 network_summary.py
