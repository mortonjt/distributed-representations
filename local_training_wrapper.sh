#!/bin/bash

# This is its own script because we want `find_by_rank` to execute in local node context.

RANK=`python /mnt/home/cchandler/Development/shaggy-dog/find_my_rank.py $SAVE_DIR/ranklist`

echo "My rank is $RANK"
python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$RANK \
    --master_addr="$MASTER" \
    --master_port="$MPORT" \
    $(which fairseq-train) $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 100 \
    --arch gert \
    --tensorboard-logdir $TB_DIR \
    --bpe gpt2 --memory-efficient-fp16 \
    --num-workers 1 \
    --save-interval-updates 300 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --distributed-port $MPORT \
    --distributed-world-size $WORLDSIZE \
    --save-dir $SAVE_DIR \
    --all-gather-list-size 1048576
    

