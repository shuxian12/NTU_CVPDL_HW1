testing_folder=/tmp2/shuxian/cvpdl/hw1_109504502/Co-DETR/work_dirs/co_dino_5scale_swin_large_16e_o365tococo_hw1
# checkpoint=$testing_folder/epoch_14.pth
CONFIG_FILE=$testing_folder/../test/co_dino_5scale_swin_large_16e_o365tococo_hw1.py
# CUDA_VISIBLE_DEVICES=3 python tools/test.py \
#     $CONFIG_FILE \
#     $checkpoint \
#     --work-dir /tmp2/shuxian/cvpdl/hw1_109504502/Co-DETR/work_dirs/test \
#     --out /tmp2/shuxian/cvpdl/hw1_109504502/Co-DETR/work_dirs/test/test_results.pkl \
#     --eval bbox

for i in 9 11 12 13 14 15 16
do
    echo "Testing epoch $i"
    checkpoint=$testing_folder/epoch_$i.pth
    CUDA_VISIBLE_DEVICES=3 python tools/test.py \
        $CONFIG_FILE \
        $checkpoint \
        --work-dir /tmp2/shuxian/cvpdl/hw1_109504502/Co-DETR/work_dirs/test \
        --out /tmp2/shuxian/cvpdl/hw1_109504502/Co-DETR/work_dirs/test/test_${i}_results.pkl \
        --eval bbox > /tmp2/shuxian/cvpdl/hw1_109504502/Co-DETR/work_dirs/test/test_${i}_results.log
done