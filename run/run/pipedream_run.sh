#!/bin/bash
# Launch all pipedream processes for 1 node.
# Is called from run_template.sh

usage() {
    echo "$0 usage:" && grep " .)\ #" $0; exit 0;
}
[ $# -eq 0 ] && usage

while getopts "m:n:t:l:c:b:w:s:i:o:a:d:e:r:h" arg; do
    case $arg in
        m) # Master node ip
            master=${OPTARG}
            ;;
        n) # Model name
            mname=${OPTARG}
            ;;
        t) # total number of gpus
            totalgpu=${OPTARG}
            ;;
        l) # Gpus per node
            localgpus=${OPTARG}
            ;;
        c) # Parallel configuration
            config=${OPTARG}
            ;;
        b) # Backend
            backend=${OPTARG}
            ;;
        w) # Number of cpu workers per gpu
            workers=${OPTARG}
            ;;
        s) # Use synthetic data set
            synth=${OPTARG}
            ;;
        i) # Node index
            nodeindex=${OPTARG}
            ;;
        o) # Print log after every X batches
            loginter=${OPTARG}
            ;;
        a) # Batchsize
            batchsize=${OPTARG}
            ;;
        d) # Path to data
            datadir=${OPTARG}
            ;;
        e) # Number of epochs
            epochs=${OPTARG}
            ;;
        r) # Use highres dataset
            highres=${OPTARG}
            ;;
        h | *) # Display help.
            usage
            exit 0
            ;;
    esac
done

#--------------------------------------------------------------------------------------
source /etc/profile

module purge
module load 2019
module load Python/3.6.6-foss-2019b
module load CMake/3.12.1-GCCcore-8.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243

# All python pip libaries have already been installed in the main script
source ~/.envs/env_recdistr_pipedream/bin/activate

cd $datadir
cd slurm*
cd scratch
new_datadir=$(pwd)

#--------------------------------------------------------------------------------------
cd ~/dnn-benchmark-suite/pipedream-fork/runtime/image_classification
echo "totalgpu=${totalgpu} | model=${mname}"
echo "Launch processes from node ${nodeindex}. Master is ${master}"
for (( id=0; id<$localgpus; id++ ))
do
    echo "Node ${nodeindex}. Launch proc: global $(($nodeindex * $localgpus + $id)) | local ${id}"
    python3 main_with_runtime.py                                        \
        --module "models.${mname}.gpus=${totalgpu}"                     \
        -b $batchsize                                                   \
        --data_dir $new_datadir                                         \
        --rank $(($nodeindex * $localgpus + $id))                       \
        --local_rank $id                                                \
        --master_addr $master                                           \
        --config_path "models/${mname}/gpus=${totalgpu}/${config}.json" \
        --distributed_backend $backend                                  \
        --workers $workers                                              \
        --epochs $epochs                                                \
        --print-freq $loginter                                          \
        --num_ranks_in_server $localgpus                                \
        --synthetic_data $synth                                         \
        --highres $highres                                              &
done

echo "Wait ${nodeindex}"
wait
echo "Finished ${nodeindex}"
cd ~
