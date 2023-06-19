
MASTER_PORT=9999 \
MASTER_ADDR=localhost \
RANK=0 \
WORLD_SIZE=1 \
LOCAL_RANK=0 \
python /home/mike/Workspaces/aurmr_hitl/src/VideoMAE/run_class_finetuning.py \
            --finetune /data/aurmr/grip/models/ss2/vit_s_pretrain/checkpoint.pth \
            --nb_classes 2 \
            --batch_size 10 \
            --num_sample 2 \
            --input_size 224 \
            --short_side_size 224 \
            --epochs 30 \
            --model vit_small_patch16_224 \
            --input_size 224 \
            --data_path /data/aurmr/grip/datasets/grasp_success_classifier \
            --data_set grip \
            --output_dir /data/aurmr/grip/models/ss2/grip_binary_finetune \
            --num_frames 16 \
            --opt adamw \
            --lr 1e-3 \
            --layer_decay 0.7 \
            --opt_betas 0.9 0.999 \
            --weight_decay 0.05 \
            --epochs 40 \
            --test_num_segment 2 \
            --test_num_crop 3 \
            --enable_deepspeed \
            --log_dir ./logs