#!/bin/bash
#
#SBATCH --job-name=roberta
#SBATCH --output=dist_stdout.txt
#Add SBATCH back if you'd like to be emailed/notified
# --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
# --mail-user=jmorton@flatironinstitute.org
#
#--ntasks=4
#--cpus-per-task 40
#SBATCH --time=100:00:00
#SBATCH -N 2
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH -c 25
#SBATCH --exclusive

# see https://www.glue.umd.edu/hpcc/help/software/pytorch.html#distrib
# Replace with your own virtualenv controls.
source /mnt/home/cchandler/.bashrc
conda activate shaggy

module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
module load nccl/2.4.2-cuda-10.1

export TOTAL_UPDATES=100000    # Total number of training steps
export WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
export PEAK_LR=0.0006          # Peak learning rate, adjust as needed
export TOKENS_PER_SAMPLE=1024  # Max sequence length
export MAX_POSITIONS=1024      # Num. positional embeddings (usually same as above)
export MAX_SENTENCES=1         # Number of sequences per batch (batch size)
export UPDATE_FREQ=8           # Increase the batch size 8x
export DATA_DIR=/mnt/home/mgt/data/uniref50
export SAVE_DIR=/mnt/home/cchandler/ceph/shaggy/save
export TB_DIR=/mnt/home/cchandler/ceph/shaggy/tb

WORKERS=`scontrol show hostnames $SLURM_JOB_NODELIST`
# I got pytorch distributed notes from
# 1. http://hpcc.umd.edu/hpcc/help/software/pytorch.html
# 2. http://aaronsplace.co.uk/blog/2018-12-08-pytorch-distributed-learning.html

#Make sure this node (MASTER) comes first
HOSTLIST="$WORKERS"

#https://unix.stackexchange.com/a/132524
#It's just easier to use a fixed port in our environment
export MPORT=4664

#Nproc per node needs to match the GPU count
export NPROC_PER_NODE=4
export WORLDSIZE=$(( NPROC_PER_NODE * SLURM_JOB_NUM_NODES ))

export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG=WARN

echo "Current world size $WORLDSIZE"
echo "Current number of nodes $SLURM_JOB_NUM_NODES"
echo "Current hostlist $HOSTLIST"

# Write the nodelist with expected ranks
echo $HOSTLIST | python /mnt/home/cchandler/Development/shaggy-dog/node_to_rank_writer.py $SAVE_DIR/ranklist

RANK=0
# This loop isn't really needed, but it made it easier to pull out the 'master' node
for node in $HOSTLIST
do
    if [ "$RANK" -eq "0" ]; then
        export MASTER=$node
        echo "Designating $node as master"
    fi
    RANK=$((RANK+1))
done
srun ./local_training_wrapper.sh
wait
