#!/bin/bash
benchmark="mnist"
framework="pytorch"
gpus=1
nodes=1
loginterval=25
modelname="all"
queue="gpu_short"
synthetic=true

usage() {
    echo "$0 usage:" && grep " .)\ #" $0; exit 0;
}
[ $# -eq 0 ] && usage

while getopts "b:f:g:n:p:m:q:sh" arg; do
    case $arg in
        b) # Benchmark: mnist, cifar10, imagenet, highres (synthetic only), all. Default: mnist
            benchmark=${OPTARG}
            ;;
        f) # Framework: pytorch, horovod, gpipe, pipedream, all. Default: pytorch
            framework=${OPTARG}
            ;;
        g) # Nr. GPUs per node. Default: 1
            gpus=${OPTARG}
            ;;
        n) # Nr. of nodes. Default: 1
            nodes=${OPTARG}
            ;;
        p) # Print statistics every X batches. Default: 25
            loginterval=${OPTARG}
            ;;
        m) # Model name: resnet18, resnet50, resnet152, vgg11, vgg16, mobilenetv2, exp2 (resnet50, vgg16, mobilenetv2), all. Default: all
            modelname=${OPTARG}
            ;;
        q) # Queue name: gpu (5d), gpu_short (1h), gpu_titanrtx (5d), gpu_titanrtx_short (1h). Default: gpu_short
            queue=${OPTARG}
            ;;
        s) # Disable synthetic data generation, use real data (will take much longer)
            synthetic=false
            ;;
        h | *) # Display help.
            usage
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------------------------------------------
# Change user parameters based on special cases
if [[ "$nodes" -gt "1" && ("$framework" == "pytorch" || "$framework" == "gpipe") ]]; then
    echo -e "\n\nOnly Horovod and Pipedream support multi-node execution. Continuing with 1 node.\n\n"
    nodes=1
fi

if [[ ("$modelname" == "resnet152" || "$modelname" == "all") && ("$framework" == "pipedream" || "$framework" == "all") ]]; then
    echo -e "\n\nResnet-152 is disabled for Pipedream.\n\n"
    if [[ "$modelname" == "resnet152" && "$framework" == "pipedream" ]]; then
        echo "Only executing resnet152 with pipedream, so stopping execution."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------------------------------------------
if [ "$queue" == "gpu" ] || [ "$queue" == "gpu_short" ]; then
    cpus_per_gpu=$((12 / gpus))
elif [ "$queue" == "gpu_titanrtx" ] || [ "$queue" == "gpu_titanrtx_short" ]; then
    cpus_per_gpu=$((24 / gpus))
fi

if [ "$queue" == "gpu_short" ] || [ "$queue" == "gpu_titanrtx_short" ]; then
    queue_time="1:00:00"
elif [ "$queue" == "gpu" ] || [ "$queue" == "gpu_titanrtx" ]; then
    queue_time="5-00:00:00"
fi

# ---------------------------------------------------------------------------------------------------------------
# Create output files
cd ~/dnn-benchmark-suite
mkdir -p out
date=`date +%Y-%m-%d_%H-%M-%S`
dir="${HOME}/dnn-benchmark-suite/out/${date}"
mkdir $dir

info="${dir}/info.txt"
output_file="${dir}/slurm.out"

# Write info about this run to dir
echo -e "Benchmark      $benchmark"      > "$info"
echo -e "Framework      $framework"     >> "$info"
echo -e "GPUs / node    $gpus"          >> "$info"
echo -e "Nodes          $nodes"         >> "$info"
echo -e "Log interval   $loginterval"   >> "$info"
echo -e "Model name     $modelname"     >> "$info"
echo -e "Queue          $queue"         >> "$info"
echo -e "Use synthetic  $synthetic"     >> "$info"

# ---------------------------------------------------------------------------------------------------------------
cd ~/dnn-benchmark-suite/run/run

./run_template.sh $benchmark $framework $gpus $nodes $cpus_per_gpu $modelname \
                  $queue_time $output_file $queue                             \
                  $loginterval $synthetic
