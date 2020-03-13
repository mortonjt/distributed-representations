#!/bin/bash
#
#SBATCH --job-name=roberta
#SBATCH --output=dist_stdout.txt
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#
#SBATCH --ntasks=4
#SBATCH --cpus-per-task 40
#SBATCH --time=100:00:00
#SBATCH -N 2
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# see https://www.glue.umd.edu/hpcc/help/software/pytorch.html#distrib

# ip=`curl ifconfig.me`

module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
module load nccl/2.4.2-cuda-10.1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

# TODO: Change environment if necessary
source ~/venvs/roberta/bin/activate

TOTAL_UPDATES=100000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0006          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024  # Max sequence length
MAX_POSITIONS=1024      # Num. positional embeddings (usually same as above)
MAX_SENTENCES=1         # Number of sequences per batch (batch size)
UPDATE_FREQ=8           # Increase the batch size 8x
DATA_DIR=/mnt/home/mgt/data/uniref50
SAVE_DIR=/mnt/home/jmorton/research/gert/roberta-checkpoints/uniref50-test
TB_DIR=/mnt/home/jmorton/research/gert/roberta-checkpoints/tensorboard

OMP_NUM_THREADS=10
echo `which python`

NPROC_PER_NODE=10
COMMAND="$(which fairseq-train) $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format tqdm --log-interval 1 \
    --ddp-backend=no_c10d \
    --fix-batches-to-gpus \
    --bpe gpt2 --memory-efficient-fp16 \
    --save-interval-updates 10 \
    --keep-interval-updates 3 \
    --save-dir $SAVE_DIR"

MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
# I got pytorch distributed notes from
# 1. http://hpcc.umd.edu/hpcc/help/software/pytorch.html
# 2. http://aaronsplace.co.uk/blog/2018-12-08-pytorch-distributed-learning.html


#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER $SLAVES"

#Get a random unused port on this host(MASTER) between 2000 and 9999
#First line gets list of unused ports
#2nd line restricts between 2000 and 9999
#3rd line gets single random port from the list
#MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
#grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
#sort | uniq | shuf`

#https://unix.stackexchange.com/a/132524
MPORT=`python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`

echo $COMMAND
echo $MPORT
echo $MASTER
echo $SLURM_JOB_NUM_NODES
#Launch the pytorch processes, first on master (first in $HOSTLIST) then
#on the slaves
RANK=0
#port=`echo $MPORT | awk '{print $'$((RANK+1))'}'`
#port=4000
#echo $port
NPROC_PER_NODE=2
SLURM_JOB_NUM_NODES=4
for node in $HOSTLIST
do
    #ssh -q $node \
    srun python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$RANK --master_addr="$MASTER" --master_port="$MPORT" \
    $(which fairseq-train) $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 100 \
    --ddp-backend=no_c10d \
    --arch gert \
    --tensorboard-logdir $TB_DIR \
    --bpe gpt2 --memory-efficient-fp16 \
    --num-workers $NPROC_PER_NODE \
    --save-interval-updates 300 \
    --save-dir $SAVE_DIR &
    RANK=$((RANK+1))
done
wait
