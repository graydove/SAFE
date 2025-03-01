
GPU_NUM=2
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12588

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

train_datasets=(
    "data/datasets/train_ForenSynths/train" \
)
eval_datasets=(
    "data/datasets/train_ForenSynths/val" \
)

MODEL="SAFE"

for train_dataset in "${train_datasets[@]}" 
do
    for eval_dataset in "${eval_datasets[@]}" 
    do

        current_time=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_PATH="results/$MODEL/$current_time"
        mkdir -p $OUTPUT_PATH

        python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
            --input_size 256 \
            --transform_mode 'crop' \
            --model $MODEL \
            --data_path "$train_dataset" \
            --eval_data_path "$eval_dataset" \
            --save_ckpt_freq 1 \
            --batch_size 256 \
            --blr 1e-4 \
            --weight_decay 0.01 \
            --warmup_epochs 1 \
            --epochs 200 \
            --num_workers 16 \
            --output_dir $OUTPUT_PATH \
            --seed 3 \
        2>&1 | tee -a $OUTPUT_PATH/log_train.txt

    done
done