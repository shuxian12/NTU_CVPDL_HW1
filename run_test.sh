testing_folder=work_dirs/co_dino_5scale_swin_large_16e_o365tococo_hw1
checkpoint=$testing_folder/epoch_14.pth
CONFIG_FILE=$testing_folder/co_dino_5scale_swin_large_16e_o365tococo_hw1.py

CUDA_VISIBLE_DEVICES=3 python tools/test.py \
    $CONFIG_FILE \
    $checkpoint \
    --work-dir work_dirs/test \
    --out work_dirs/test/test_results.pkl \
