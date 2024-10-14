# CVPDL HW1 - Object Detection

> Name: 陳紓嫻
> Student ID: R13944014

## Environment

- OS: Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 3090 (24GB)
- Python: 3.7.12
- PyTorch: 1.11.0
- cuda: 11.3

## Installation

### Clone the repository

```bash
git clone https://github.com/shuxian12/NTU_CVPDL_HW1.git
cd NTU_CVPDL_HW1
```

### Install dependencies using conda

1. Create a new conda environment and activate it
    ```bash
    conda create -n co-detr python=3.7
    conda activate co-detr
    ```

2. Install dependencies
    ```bash
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt
    pip install opencv-python==4.10.0
    ```
3. Other dependencies
    > by [Co-DETR](https://github.com/Sense-X/Co-DETR/tree/main?tab=readme-ov-file) \
    > We implement Co-DETR using MMDetection V2.25.3 and MMCV V1.5.0. The source code of MMdetection has been included in this repo and you only need to build MMCV following [official instructions](https://github.com/open-mmlab/mmcv/tree/v1.5.0#installation). We test our models under `python=3.7.11,pytorch=1.11.0,cuda=11.3`. Other versions may not be compatible.
    
    ```bash
    pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
    ```

## Run the code

### Dataset
1. After downloading the dataset, you should covert the dataset to COCO annotation format.
    * Change the dataset path in the `covert_dataset_to_annotaions.py` file.
    ```python
    mode = #'train' or 'test'
    input_dir = f'your_dataset_root_folder/cvpdl_hw1/{mode}/'
    output_dir = # 'your_output_dir'
    # -> your output annotation file will be like {mode}.json
    ```
    * Run the script
    ```bash
    python covert_dataset_to_annotaions.py
    ```
    * Put the output file `val.json`, `train.json`, `test.json` in the `your_dataset_root_folder/cvpdl_hw1/annotations` folder.
    * Take the cvpdl_hw1 dataset as an example, the folder structure should be like this:
    ```bash
    cvpdl_hw1/
    ├── annotations/
    │   ├── test.json
    │   ├── train.json
    │   └── val.json
    ├── train/
    │   ├──images/
    │   └──labels/
    ├── valid/
    │   ├──images/
    │   └──labels/
    └── test/
        └──images/
    ```
2. Then, you should edit the dataset path in the `projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo_hw1.py` file.
    * In line 162
    ```python
    data_root='/path/to/your/dataset' #e.g. 'your/path/cvpdl_hw1/' or absolute path
    ```

### Pretrained Model

* Dowload link: [co_dino_5scale_swin_large_16e_o365tococo.pth](https://drive.google.com/file/d/1ffDz9lGNAjEF7iXzINZezZ4alx6S0KcO/view?usp=share_link)
* Put the pretrained model in the root folder.
```bash
NTU_CVPDL_HW1/
├── co_dino_5scale_swin_large_16e_o365tococo.pth
├── ...
```

### Train
```bash
. run_train.sh
```
* The output log and checkpoints will be saved in the `work_dirs/co_dino_5scale_swin_large_16e_o365tococo_hw1` folder.

### Test
1. Change the `checkpoint` path in the `run_test.sh` file, and the default path is the best epoch 14,(you can change it to other epochs).
    ```bash
    # since the best epoch is 14
    checkpoint=$testing_folder/epoch_14.pth
    ```
2. Run the test script, and the result will be saved in the `work_dirs/test/results.pkl` file.
    
    ```bash
    . run_test.sh
    ```
  * If you want to change the output path, you can edit the **`--out`** parameter in `run_test.sh` file.
3. If you want to get the valid dataset predicted result, you should change the `test` of `data` in the `work_dirs/co_dino_5scale_swin_large_16e_o365tococo_hw1/co_dino_5scale_swin_large_16e_o365tococo_hw1.py` file. 
    > The eaiest way is to copy the `val` config and change the `ann_file` and `img_prefix` to the valid dataset path.
    ```python
    test=dict(
        type='CocoDataset',
        ann_file= # 'same as the val config',
        img_prefix='{your dataset of valid folder}/images/',
        pipeline=...
    )
    ```
4. Then run **step 2** again.

5. Covert the result to the submission format.
    ```sh
    python covert_result_to_submission.py \ 
          --config work_dirs/co_dino_5scale_swin_large_16e_o365tococo_hw1/co_dino_5scale_swin_large_16e_o365tococo_hw1.py \
          --input_pkl work_dirs/test/results.pkl  --output work_dirs/test/results.json
    ```
