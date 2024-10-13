# The COCO dataset is saved as a JSON file in the output directory.

import json
import os
from PIL import Image

# Set the paths for the input and output directories
mode = 'train' # 'train' or 'test' or 'valid'
input_dir = f'your_dataset_root_folder/cvpdl_hw1/{mode}/'
output_dir = 'your_output_dir'
if mode == 'valid':
    annotations_file = 'val.json'
else:
    annotations_file = f'{mode}.json'

# Define the categories for the COCO dataset
categories = [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "ear"},
    {"id": 2, "name": "ear-mufs"},
    {"id": 3, "name": "face"},
    {"id": 4, "name": "face-guard"},
    {"id": 5, "name": "face-mask"},
    {"id": 6, "name": "foot"},
    {"id": 7, "name": "tool"},
    {"id": 8, "name": "glasses"},
    {"id": 9, "name": "gloves"},
    {"id": 10, "name": "helmet"},
    {"id": 11, "name": "hands"},
    {"id": 12, "name": "head"},
    {"id": 13, "name": "medical-suit"},
    {"id": 14, "name": "shoes"},
    {"id": 15, "name": "safety-suit"},
    {"id": 16, "name": "safety-vest"}
]

# Define the COCO dataset dictionary
coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}

# Loop through the images in the input directory
img_path = os.path.join(input_dir, "images")
id = 0
img_path_list = [p for p in os.listdir(img_path) if not p.endswith('.DS_Store')]

for image_file in img_path_list:
    # Load the image and get its dimensions
    image_path = os.path.join(img_path, image_file)
    image = Image.open(image_path)
    width, height = image.size
    
    # Add the image to the COCO dataset
    image_dict = {
        "id": id, #int(image_file.split('.')[0]),
        "width": width,
        "height": height,
        "file_name": image_file
    }
    coco_dataset["images"].append(image_dict)
    
    if mode != 'test':
        # Load the bounding box annotations for the image
        with open(os.path.join(input_dir, "labels",f'{image_file.split(".")[0]}.txt')) as f:
            annotations = f.readlines()
        
        # Loop through the annotations and add them to the COCO dataset
        for ann in annotations:
            category_id = int(ann.strip().split()[0])
            x, y, w, h = map(float, ann.strip().split()[1:])
            x_min, y_min = (x - w / 2) * width, (y - h / 2) * height
            x_max, y_max = (x + w / 2) * width, (y + h / 2) * height
            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": id,  # int(image_file.split('.')[0]),
                "category_id": category_id,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],   # [x_min, y_min, w, h],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)
    id += 1

# Save the COCO dataset to a JSON file
with open(os.path.join(output_dir, annotations_file), 'w') as f:
    json.dump(coco_dataset, f, indent=4)
