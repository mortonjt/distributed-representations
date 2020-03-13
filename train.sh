0;95;0c#!/bin/bash
#
#SBATCH --job-name=roberta
#SBATCH --output=stdout.txt
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#
#SBATCH --ntasks=40
#SBATCH --time=200:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# see https://www.glue.umd.edu/hpcc/help/software/pytorch.html#distrib
# may want to launch at pcn-7-12

ip=`curl ifconfig.me`

module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
module load nccl/2.4.2-cuda-10.1
#export NCCL_DEBUG=WARN
#export NCCL_DEBUG_SUBSYS=ALL
#cd /simons/scratch/jmorton/mgt

# TODO: Change environment if necessary
source ~/venvs/roberta/bin/activate

# TODO: These 3 paths below are critical, you may want to change those
DATA_DIR=/mnt/home/mgt/data/uniref50
SAVE_DIR=/mnt/home/jmorton/research/gert/roberta-checkpoints/uniref50-test
TB_DIR=/mnt/home/jmorton/research/gert/roberta-checkpoints/tensorboard

mkdir -p $SAVE_DIR
mkdir -p $TB_DIR

TOTAL_UPDATES=100000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0009          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024  # Max sequence length
MAX_POSITIONS=1024      # Num. positional embeddings (usually same as above)
MAX_SENTENCES=4         # Number of sequences per batch (batch size)
UPDATE_FREQ=8           # Increase the batch size by fold
# OMP_NUM_THREADS=10
echo `which python`

NPROC_PER_NODE=40
# Roberta 'gert' model
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
    --save-dir $SAVE_DIR

# Roberta xxs-gert model
# $(which fairseq-train) $DATA_DIR \
#     --task masked_lm --criterion masked_lm \
#     --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
#     --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
#     --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
#     --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
#     --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
#     --max-update $TOTAL_UPDATES --log-format simple --log-interval 100 \
#     --ddp-backend=no_c10d \
#     --arch gert \
#     --bpe gpt2 --memory-efficient-fp16 \
#     --num-workers $NPROC_PER_NODE \
#     --save-interval-updates 10000 \
#     --skip-invalid-size-inputs-valid-test \
#     --save-dir $SAVE_DIR
