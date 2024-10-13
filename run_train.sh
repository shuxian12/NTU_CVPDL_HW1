# CONFIG_FILE=projects/configs/co_dino_vit/co_dino_5scale_vit_large_hw1.py
CONFIG_FILE=projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo_hw1.py
python tools/train.py \
    $CONFIG_FILE \
    --resume-from /tmp2/shuxian/cvpdl/hw1_109504502/Co-DETR/work_dirs/co_dino_5scale_swin_large_16e_o365tococo_hw1/epoch_1.pth
# CONFIG=projects/configs/co_dino_vit/co_dino_5scale_vit_large_hw1.py
# GPUS=3

# PORT=${PORT:-29500}

# # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# # echo $PYTHONPATH
# device=4,5
# CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     tools/train.py $CONFIG
