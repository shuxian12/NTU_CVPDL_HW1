import sys
import argparse 
# sys.path.append('../../')

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root, compat_cfg

parser = argparse.ArgumentParser()
# parser.add_argument('--epoch', type=int, default=16)
parser.add_argument('--config', type=str, 
                    default='work_dirs/co_dino_5scale_swin_large_16e_o365tococo_hw1/co_dino_5scale_swin_large_16e_o365tococo_hw1.py')
parser.add_argument('--output', type=str, default='test.json')
parser.add_argument('--input_pkl', type=str, default='test.pkl')

args = parser.parse_args()

cfg = Config.fromfile(args.config)

# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)

# update data root according to MMDET_DATASETS
update_data_root(cfg)

cfg = compat_cfg(cfg)
dataset = build_dataset(cfg.data.test)
print("Loading dataset done...")
import pickle
outputs = pickle.load(open(args.input_pkl, 'rb'))

dataset.img_ids = dataset.coco.get_img_ids()
dataset.data_infos = ['' for _ in range(len(dataset.coco.get_img_ids()))]
print(f"dataset.img_ids: {len(dataset.img_ids)}, dataset.data_infos: {len(dataset.data_infos)}, outputs: {len(outputs)}")
json_results = dataset._det2json(outputs)
print("Setting up dataset done...")
# print(json_results[0], len(json_results))

import json
anno = json.load(open(cfg.data.test.ann_file))
# img_id_to_name = anno['images']
img_id_to_name = {img['id']: img for img in anno['images']}
'''
{
    img_name: {
        'bbox': [[x1, y1, x2, y2], ...],
        'score': [score, ...],
        'label': [label, ...]
    }
}
'''

output_predictions = {
    _img['file_name']: {
        'boxes': [],
        'labels': [],
        'scores': [],
    } for _img in img_id_to_name.values()
}

for res in json_results:
    img_name = img_id_to_name[res['image_id']]['file_name']
    x, y, w, h = res['bbox']
    x1, y1, x2, y2 = x, y, x + w, y + h
    output_predictions[img_name]['boxes'].append([x1, y1, x2, y2])
    output_predictions[img_name]['scores'].append(res['score'])
    output_predictions[img_name]['labels'].append(res['category_id'])

print("Setting up output_predictions done...")
import json
with open(args.output, 'w') as f:
        json.dump(output_predictions, f, indent=4)
        
        
'''
# setting score threshold
DATA_PATH = 'json submission path'

data = json.load(open(DATA_PATH, 'r'))

new_data = {}

for img_name, pred in data.items():
    boxes = np.array(pred['boxes'])
    labels = np.array(pred['labels'])
    scores = np.array(pred['scores'])
    
    mask = scores > 0.3
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    new_data[img_name] = {
        'boxes': boxes.tolist(),
        'labels': labels.tolist(),
        'scores': scores.tolist()
    }
    
with open('valid_R13944014.json', 'w') as f:
    json.dump(new_data, f, indent=4)
'''
