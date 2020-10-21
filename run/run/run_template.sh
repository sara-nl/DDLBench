#!/bin/bash
sbatch << EOT
#!/bin/bash
#SBATCH -J allperf
#SBATCH -t $7
#SBATCH -N $4
#SBATCH --ntasks-per-node $3
#SBATCH -p $9
#SBATCH -c $5
#SBATCH -o $8
#SBATCH -e $8
#SBATCH --gres=gpu:$3

###################################################################################################
# Option                        | Values                             | #SBATCH only | Export only |
###################################################################################################
# $1    Benchmark name          | mnist, imagenet, cifar10           |              |             |
# $2    Framework               | pytorch, horovod, gpipe, pipedream |              |             |
# $3    Nr. GPUs per node       | 1 >=                               |              |             |
# $4    Nr. of nodes            | 1 >=                               |              |             |
# $5    Nr. CPUs per GPU        | 1 >=                               |              |             |
# $6    Model name (imagenet)   | resnet50, vgg16 etc                |              |             |
# $7    Queue time              | 00:01 >=                           | Yes          |             |
# $8    Output file             | dir path + output file             | Yes          |             |
# $9    Queue name              | gpu_short (1 node), gpu            | Yes          |             |
# $10   Print batch interval    | 1 >=                               |              | Yes         |
# $11   Use synthetic           | 0 (true), 1 (false)                |              |             |
###################################################################################################

# Usage: contains list item
contains() {
    [[ \$1 =~ (^|[[:space:]])\$2($|[[:space:]]) ]] && return 0 || return 1
}

cd ~

#--------------------------------------------------------------------------------------
# Parse arguments to list
benchmark=$1
if [ "$1" == "all" ]; then
    # Highres is a synthetic dataset
    if ${11}; then
        benchmark="mnist cifar10 imagenet highres"
    else
        benchmark="mnist cifar10 imagenet"
    fi
fi

framework=$2
if [ "$2" == "all" ]; then
    framework="pytorch horovod gpipe pipedream"
    
    # Not wasting multi-node run on pytorch (single gpu) and gpipe (single node, multi gpu)
    if [ "$4" -gt "1" ]; then
        echo "multi-node only supports horovod and pipedream"
        framework="horovod pipedream"
    fi
fi

modelname=$6
if [ "$6" == "all" ]; then
    modelname="resnet18 resnet50 resnet152 vgg11 vgg16 mobilenetv2"
elif [ "$6" == "exp2" ]; then
    modelname="resnet50 vgg16 mobilenetv2"
fi

#--------------------------------------------------------------------------------------
# First set env variables

export CORES_GPU=$5
export EPOCHS=3
export LOGINTER=${10}
export DATADIR="${TMPDIR}"

#--------------------------------------------------------------------------------------
# Get jobs to nodes / gpus maps
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile
host=\$(cat hostfile)
hosts=( \$host )
hostsarr=( \$host )
hostsarr1=( \$host )
for ((i=0;i<\${#hostsarr[@]};i++)); do
    hostsarr[i]="\${hostsarr[i]}:${3}"
    hostsarr1[i]="\${hostsarr1[i]}:1"
done
harr1=\$(IFS=, ; echo "\${hostsarr1[*]}")       # 1 worker per node, for data copy
harr=\$(IFS=, ; echo "\${hostsarr[*]}")         # #gpus workers per node, for execution
rm hostfile

#--------------------------------------------------------------------------------------
# Order of execution: <pytorch, horovod, gpipe>, <pipedream>
mkdir -p .envs

if contains "\$framework" "pytorch" || contains "\$framework" "horovod" || contains "\$framework" "gpipe"; then
    module purge
    module use ~/environment-modules-lisa
    module load 2019
    module load Python/3.6.6-foss-2019b

    module unload GCC
    module unload GCCcore
    module unload binutils
    module unload zlib
    module unload compilerwrappers

    module load NCCL/2.5.6-CUDA-10.1.243
    module load cuDNN/7.6.5.32-CUDA-10.1.243
    module load OpenMPI/3.1.4-GCC-8.3.0
    module load libyaml/0.2.1-GCCcore-8.3.0

    export MPICC=mpicc
    export MPICXX=mpicxx

    python3 -m venv ~/.envs/env_recdistr
    source ~/.envs/env_recdistr/bin/activate

    pip3 install pyyaml

    # We want cuda 10.1 torch / torchvision
    pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

    # Install misc python packages
    cd ~/dnn-benchmark-suite/run
    pip3 install -r requirements.txt
    cd ~

    #--------------------------------------------------------------------------------------
    if ${11}; then
        # Generate synthetic data once
        echo "Generate synthetic data"
        if [ $4 -gt 1 ]; then
            mpirun -N 1 -H \$harr1 python3 ~/dnn-benchmark-suite/benchmark/generate_synthetic_data.py \$benchmark
        else
            python3 ~/dnn-benchmark-suite/benchmark/generate_synthetic_data.py \$benchmark
        fi

        synth=0
    else
        # Copy the real data
        cd "$TMPDIR"

        if contains "\$benchmark" "mnist"; then
            echo "Copy real mnist data"
            mpirun -N 1 -H \$harr1 tar zxf ~/benchmark/mnist/MNIST.tar.gz
        fi
        if contains "\$benchmark" "cifar10"; then
            echo "Copy real cifar10 data"
            mpirun -N 1 -H \$harr1 tar zxf ~/benchmark/cifar10/cifar-10-python.tar.gz
        fi
        if contains "\$benchmark" "imagenet"; then
            # Use MPIcopy here as the other two data sets are very small (~70mb) while this one is ~150gb    
            module load mpicopy

            echo "Copying imagenet, this will take some time: $(date +'%T')"
            mpicopy /nfs/managed_datasets/imagenet/train
            echo "Copied train over: $(date +'%T')"
            mpicopy /nfs/managed_datasets/imagenet/val
            echo "Done with copying imagenet: $(date +'%T')"
        fi
        synth=1
        cd ~
    fi

    #--------------------------------------------------------------------------------------
    # Already install horovod here as the performance predictor will use it early
    if contains "\$framework" "horovod"; then
        echo "Install horovod"
        export HOROVOD_CUDA_HOME=$CUDA_HOME
        export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
        export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
        export HOROVOD_NCCL_HOME=$EBROOTNCCL
        export HOROVOD_GPU_ALLREDUCE=NCCL
        export HOROVOD_GPU_BROADCAST=NCCL
        export HOROVOD_WITH_PYTORCH=1

        pip3 install --no-cache-dir horovod
    fi
fi

cd ~/dnn-benchmark-suite

#--------------------------------------------------------------------------------------
if contains "\$framework" "pytorch"; then
    for model in \$modelname; do
        if contains "\$benchmark" "mnist"; then
            export BATCH_SIZE=128
            echo "pytorch - mnist - \${model} - batch=\${BATCH_SIZE}"
            python3 "benchmark/mnist/mnist_pytorch.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "cifar10"; then
            export BATCH_SIZE=64
            echo "pytorch - cifar10 - \${model} - batch=\${BATCH_SIZE}"
            python3 "benchmark/cifar10/cifar10_pytorch.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "imagenet"; then
            export BATCH_SIZE=32
            echo "pytorch - imagenet - \${model} - batch=\${BATCH_SIZE}"
            python3 "benchmark/imagenet/imagenet_pytorch.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "highres"; then
            export BATCH_SIZE=32
            echo "pytorch - highres - \${model} - batch=\${BATCH_SIZE}"
            python3 "benchmark/imagenet/imagenet_pytorch.py" -a \$model -s -1
        fi
    done
fi

#--------------------------------------------------------------------------------------
if contains "\$framework" "horovod"; then
    totalgpu=$(($3 * $4))
    for model in \$modelname; do
        if contains "\$benchmark" "mnist"; then
            export BATCH_SIZE=128
            echo "horovod - mnist - \${model} - batch=\${BATCH_SIZE}"
            horovodrun -np \$totalgpu -H \$harr python3 "benchmark/mnist/mnist_horovod.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "cifar10"; then
            export BATCH_SIZE=64
            echo "horovod - cifar10 - \${model} - batch=\${BATCH_SIZE}"
            horovodrun -np \$totalgpu -H \$harr python3 "benchmark/cifar10/cifar10_horovod.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "imagenet"; then
            export BATCH_SIZE=32
            echo "horovod - imagenet - \${model} - batch=\${BATCH_SIZE}"
            horovodrun -np \$totalgpu -H \$harr python3 "benchmark/imagenet/imagenet_horovod.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "highres"; then
            export BATCH_SIZE=32
            echo "horovod - highres - \${model} - batch=\${BATCH_SIZE}"
            horovodrun -np \$totalgpu -H \$harr python3 "benchmark/imagenet/imagenet_horovod.py" -a \$model -s -1
        fi
    done
fi

#--------------------------------------------------------------------------------------
if contains "\$framework" "gpipe"; then
    echo "Install gpipe"
    pip3 install torchgpipe
fi

if contains "\$framework" "gpipe"; then
    for model in \$modelname; do
        if contains "\$benchmark" "mnist"; then
            export BATCH_SIZE=128
            export MICROBATCHES=24
            echo "gpipe - mnist - \${model} - batch=\${BATCH_SIZE}"
            python3 "benchmark/mnist/mnist_gpipe.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "cifar10"; then
            export BATCH_SIZE=64
            export MICROBATCHES=32
            echo "gpipe - cifar10 - \${model} - batch=\${BATCH_SIZE}"
            python3 "benchmark/cifar10/cifar10_gpipe.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "imagenet"; then
            export BATCH_SIZE=24
            export MICROBATCHES=12
            echo "gpipe - imagenet - \${model} - batch=\${BATCH_SIZE}"
            python3 "benchmark/imagenet/imagenet_gpipe.py" -a \$model -s \${synth}
        fi
        if contains "\$benchmark" "highres"; then
            export BATCH_SIZE=4
            export MICROBATCHES=12
            echo "gpipe - highres - \${model} - batch=\${BATCH_SIZE}"
            python3 "benchmark/imagenet/imagenet_gpipe.py" -a \$model -s -1
        fi
    done
fi

#--------------------------------------------------------------------------------------
if contains "\$framework" "pipedream"; then
    module purge
    module load 2019
    module load Python/3.7.5-foss-2019b
    module load CMake/3.12.1-GCCcore-8.3.0
    module load cuDNN/7.6.5.32-CUDA-10.1.243
    module load NCCL/2.5.6-CUDA-10.1.243

    cd ~
    python3 -m venv ~/.envs/env_recdistr_pipedream
    source ~/.envs/env_recdistr_pipedream/bin/activate

    echo "Install pipedream"
    pip3 install ~/dnn-benchmark-suite/torch_pipedream/pytorch/dist/torch-1.1.0a0+828a6a3-cp37-cp37m-linux_x86_64.whl
    pip3 install torchvision==0.2.1

    cd ~/dnn-benchmark-suite/pipedream-fork
    pip3 install -r requirements.txt
    cd ~

    #--------------------------------------------------------------------------------------
    # Create symlinks to models if not already there
    model_cifar10="${HOME}/dnn-benchmark-suite/benchmark/cifar10/pytorchcifargitmodels"
    model_mnist="${HOME}/dnn-benchmark-suite/benchmark/mnist/models"
    model_target="${HOME}/dnn-benchmark-suite/pipedream-fork/profiler/image_classification/models"

    if [ ! -f "\${model_target}/mnistresnet.py" ]; then
        ln -s "\${model_mnist}/mnistresnet.py" "\${model_target}/mnistresnet.py"
    fi
    if [ ! -f "\${model_target}/mnistvgg.py" ]; then
        ln -s "\${model_mnist}/mnistvgg.py" "\${model_target}/mnistvgg.py"
    fi
    if [ ! -f "\${model_target}/mnistmobilenetv2.py" ]; then
        ln -s "\${model_mnist}/mnistmobilenetv2.py" "\${model_target}/mnistmobilenetv2.py"
    fi
    if [ ! -f "\${model_target}/cifarresnet.py" ]; then
        ln -s "\${model_cifar10}/resnet.py" "\${model_target}/cifarresnet.py"
    fi
    if [ ! -f "\${model_target}/cifarvgg.py" ]; then
        ln -s "\${model_cifar10}/vgg.py" "\${model_target}/cifarvgg.py"
    fi
    if [ ! -f "\${model_target}/cifarmobilenetv2.py" ]; then
        ln -s "\${model_cifar10}/mobilenetv2.py" "\${model_target}/cifarmobilenetv2.py"
    fi

    cd ~/dnn-benchmark-suite/run
    pip3 install -r requirements.txt
    cd ~

    pip3 install Pillow==6.1
    
    #--------------------------------------------------------------------------------------
    # Only do this if not already copied
    # Is repeated here as the script uses a different pytorch version
    if ! contains "\$framework" "pytorch" && ! contains "\$framework" "horovod" && ! contains "\$framework" "gpipe"; then
        if ${11}; then
            # Generate synthetic data once
            echo "Generate synthetic data"
            mpirun -N 1 -H \$harr1 python3 ~/dnn-benchmark-suite/benchmark/generate_synthetic_data.py \$benchmark
            synth=0
        else
            # Copy the real data
            cd "$TMPDIR"

            if contains "\$benchmark" "mnist"; then
                echo "Copy real mnist data"
                mpirun -N 1 -H \$harr1 tar zxf ~/benchmark/mnist/MNIST.tar.gz
            fi
            if contains "\$benchmark" "cifar10"; then
                echo "Copy real cifar10 data"
                mpirun -N 1 -H \$harr1 tar zxf ~/benchmark/cifar10/cifar-10-python.tar.gz
            fi
            if contains "\$benchmark" "imagenet"; then
                # Use MPIcopy here as the other two data sets are very small (~70mb) while this one is ~150gb    
                module load mpicopy

                echo "Copying imagenet, this will take some time: $(date +'%T')"
                mpicopy /nfs/managed_datasets/imagenet/train
                echo "Copied train over: $(date +'%T')"
                mpicopy /nfs/managed_datasets/imagenet/val
                echo "Done with copying imagenet: $(date +'%T')"
            fi
            synth=1
            cd ~
        fi
    fi
fi

#--------------------------------------------------------------------------------------
if contains "\$framework" "pipedream"; then
    # For some reason pipedream does not like many workers
    # The program crashes with an out-of-memory error although it completely
    # isnt out of memory. So I limit it to 3 cpus-per-gpu, which seems to work.
    workers=$5
    if [ "$5" -gt "3" ]; then
        workers=3
    fi

    for model in \$modelname; do
        for dataset in \$benchmark; do
            # Resnet152 is skipped as 'convert_graph_to_model.py' is bugged
            if contains "\$model" "resnet152"; then
                continue
            fi

            highres=0
            if [ "\$dataset" == "mnist" ]; then
                export BATCH_SIZE=512
                echo "pipedream - mnist - \${model} - batch=\${BATCH_SIZE}"
                mname="\${dataset}_\${model}"
            elif [ "\$dataset" == "cifar10" ]; then
                export BATCH_SIZE=256
                echo "pipedream - cifar10 - \${model} - batch=\${BATCH_SIZE}"
                mname="\${dataset}_\${model}"
            elif [ "\$dataset" == "imagenet" ]; then
                export BATCH_SIZE=128
                echo "pipedream - imagenet - \${model} - batch=\${BATCH_SIZE}"
                mname=\$model
            elif [ "\$dataset" == "highres" ]; then
                export BATCH_SIZE=64
                echo "pipedream - highres - \${model} - batch=\${BATCH_SIZE}"
                mname=\$model
                highres=1
            fi

            #----------------------------------------------------------------
            # Run graph profiler
            # Note: You can add learning rate / momentum etc to this
            cd ~/dnn-benchmark-suite/pipedream-fork/profiler/image_classification
            echo "Start profiler"
            CUDA_VISIBLE_DEVICES=0 python3 main.py          \
                                    -a \$mname              \
                                    -b \$BATCH_SIZE         \
                                    --data_dir $TMPDIR      \
                                    --workers \${workers}   \
                                    --epochs \$EPOCHS       \
                                    --print-freq \$LOGINTER \
                                    --synthetic_data \${synth} \
                                    --highres \${highres}

            #----------------------------------------------------------------
            # Run optimizer and extract replications per layer
            echo "Start optimizer_graph_hierachical"
            NETWORK_BANDWIDTH=5000000000                            # Ethernet 40Gbit/s = 5 GB/s
            PCIE_BANDWIDTH=32000000000                              # PCIe 3.x * 16 = ~ 32 GB/s duplex
            if [ "$9" == "gpu" ] || [ "$9" == "gpu_short" ]; then
                MEMORY_SIZE=11000000000                             # 11GB on 1080TI
            elif [ "$9" == "gpu_titanrtx" ] || [ "$9" == "gpu_titanrtx_short" ]; then
                MEMORY_SIZE=24000000000                             # 24GB on Titan RTX
            fi

            # Repeat number of machines and bandwith #nodes times if > 1 node
            if [ $4 -gt 1 ]; then
                bandwidths=( \$PCIE_BANDWIDTH \$NETWORK_BANDWIDTH ) # intra, inter
                ns=( $3 $4 )
            else
                bandwidths=( \$PCIE_BANDWIDTH )
                ns=( $3 )
            fi

            #----------------------------------------------------------------
            cd ~/dnn-benchmark-suite/pipedream-fork/optimizer
            capture=0
            multi=()
            replications=()
            mapfile -t output < <(python3 optimizer_graph_hierarchical.py                          \
                            -f "../profiler/image_classification/profiles/\${mname}/graph.txt"     \
                            -n \${ns[*]}                                                           \
                            --activation_compression_ratio 1                                       \
                            -o "\${mname}_partitioned"                                             \
                            -b \${bandwidths[*]}                                                   \
                            -s \$MEMORY_SIZE                                                       \
                            --use_memory_constraint                                                )

            for line in "\${output[@]}"
            do
                echo "\$line"
                if [[ \$line =~ "Split start, split end" ]]; then
                    capture=1
                elif [[ \$line =~ "Level 1" ]]; then
                    # Multi node: Check upper level.
                    multi=( "\${replications[@]}" )
                    replications=()
                elif [ "\$capture" == "1" ]; then
                    if [ "\$line" == "" ]; then
                        capture=0
                    else
                        replication=$(echo "\${line##* }")
                        replications+=(\$replication)
                    fi
                fi
            done

            # If not in multi node, set multiplier to 1 for all
            if [ \${#multi[@]} -eq 0 ]; then
                multi=(1)
            fi

            # Now you have 2 lists: Multi with node level and replications with per node level
            # Example: 4 nodes, 4 gpus. Multi=(3,1) and replications=(3,1,1,1,1,1)
            # Result: 0:9,1:3,2:1,3:1,4:1,5:1

            # Convert replications to input string
            stage_map=""
            stage=0
            multi_count=0
            gpu_count=0
            for i in "\${replications[@]}"
            do
                # Add a , between all stages
                if [ "\$stage_map" != "" ]; then
                    stage_map="\${stage_map},"
                fi

                # Multiply with multi node
                repl=\$((\$i * \${multi[\${multi_count}]}))

                # If #gpus == gpu_per node, jump to next top_level node multiplier (next value in multi)
                gpu_count=\$((gpu_count+i))
                if [ "\$gpu_count" == "$3" ]; then
                    gpu_count=0
                    multi_count=\$((multi_count+1))
                fi

                # Append to final string
                stage_map="\${stage_map}\${stage}:\${repl}"
                stage=\$((stage+1))
            done

            # Execute graph convert + set parameters based on type of execution model
            echo "Start convert_graph_to_model"
            totalgpu=$(($3 * $4))

            # 1 comma = 2 entries
            len=\$(echo "\${stage_map}" | awk -F"," '{print NF-1}')
            len=\$((len+1))

            if [ "\${len}" == "1" ]; then
                config='dp_conf'
                backend='nccl'
            else
                if [ "\${len}" == "\$totalgpu" ]; then
                    config='mp_conf'
                    backend='gloo'
                else
                    config='hybrid_conf'
                    backend='gloo'
                fi
            fi

            echo "\${multi[@]}"
            echo "\$config"
            echo "\$backend"
            echo "\$stage_map"

            #----------------------------------------------------------------
            python3 convert_graph_to_model.py                                           \
                -f "\${mname}_partitioned/gpus=\${totalgpu}.txt"                        \
                -n "\${mname}Partitioned"                                               \
                -a \$mname                                                              \
                -o "../runtime/image_classification/models/\${mname}/gpus=\${totalgpu}" \
                --stage_to_num_ranks \$stage_map

            #----------------------------------------------------------------
            # Run pipedream: Launch script on each node which launches #gpus_per_node processes on that node
            # Note: You can add learning rate / momentum etc to this
            echo "Start runtime"
            count=0
            for h in "\${hosts[@]}"; do
                echo "launch host \${h}"
                ssh \$h "~/dnn-benchmark-suite/run/run/pipedream_run.sh \
                    -m \${hosts[0]}    \
                    -n \$mname         \
                    -t \$totalgpu      \
                    -l $3              \
                    -c \$config        \
                    -b \$backend       \
                    -w \$workers       \
                    -s \$synth         \
                    -i \$count         \
                    -o \$LOGINTER      \
                    -a \$BATCH_SIZE    \
                    -d \$DATADIR       \
                    -e \$EPOCHS        \
                    -r \$highres       \
                    >> '${8}_\${count}'" &
                (( count++ ))
            done
            echo "Wait for nodes to finish"
            wait
            echo "All nodes finished"
            cd ~
        done
    done
fi

EOT


