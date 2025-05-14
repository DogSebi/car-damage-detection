import json
import yaml
import os
from roboflow import Roboflow
from PIL import Image, ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO

def load_dataset(api_key: str, workspace: str, project: str, format: str, version: int = 1):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    version = project.version(version)
    dataset = version.download(format)
    return dataset


def make_one_class(old_annotaion_path: str, new_annotaion_path: str):

    with open(old_annotaion_path, 'r') as f:
        data = json.load(f)

    for ann in data['annotations']:
        ann['category_id'] = 1

    data['categories'] = [{'id': 1, 'name': 'damage'}]

    with open(new_annotaion_path, 'w') as f:
        json.dump(data, f, indent=2)


def make_masks(json_path, out_mask_dir):

    os.makedirs(out_mask_dir, exist_ok=True)

    coco = COCO(json_path)
    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        mask = Image.new("L", (img_info["width"], img_info["height"]), 0)
        draw = ImageDraw.Draw(mask)

        for ann in anns:
            seg = ann["segmentation"]
            if isinstance(seg, list):
                for polygon in seg:
                    xy = list(zip(polygon[::2], polygon[1::2]))
                    draw.polygon(xy, fill=1)
            else:
                continue

        out_path = os.path.join(out_mask_dir, img_info["file_name"].replace(".jpg", ".png"))
        mask.save(out_path)

def preprocess_damage_segmentation_dataset():
    annotaions_paths = [
        (
            './car-damage-detection-1/train/_annotations.coco.json',
            './car-damage-detection-1/train/damage_single_class.json'
        ),
        (
            './car-damage-detection-1/valid/_annotations.coco.json',
            './car-damage-detection-1/valid/damage_single_class.json'
        ),
        (
            './car-damage-detection-1/test/_annotations.coco.json',
            './car-damage-detection-1/test/damage_single_class.json'
        )
    ]

    for old_annotaion_path, new_annotaion_path in annotaions_paths:
        make_one_class(old_annotaion_path, new_annotaion_path)

    print('Все повреждения объединены в один класс')

    mask_paths = [
        (
            './car-damage-detection-1/train/damage_single_class.json',
            './car-damage-detection-1/train_masks'
        ),
        (
            './car-damage-detection-1/valid/damage_single_class.json',
            './car-damage-detection-1/valid_masks'
        ),
        (
            './car-damage-detection-1/test/damage_single_class.json',
            './car-damage-detection-1/test_masks'
        )
    ]

    for json_path, out_mask_dir in mask_paths:
        make_masks(json_path, out_mask_dir)

    print('Маски созданы')
    print('Датасет для повреждений обработан')


def preprocess_yaml(yaml_path: str):

    new_names = [
        'Bumper', 'Door', 'Wheel', 'Window', 'Windshield', 'Fender', 'Grille',
        'Light', 'Hood', 'License-plate', 'Mirror', 'Quarter-panel', 'Rocker-panel',
        'Roof', 'Trunk'
    ]

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data['names'] = {i: name for i, name in enumerate(new_names)}

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print('YAML обновлён')

def preprocess_yolo_labels(label_path):

    class_map = {
        6: 0,
        7: 1,
        8: 2,
        9: 3,
        18: 7,
        20: 4,
        10: 6,
        11: 7,
        12: 8,
        13: 9,
        14: 10,
        15: 11,
        16: 12,
        17: 13,
        19: 14,
    }

    for filename in os.listdir(label_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_path, filename)
            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                tokens = line.strip().split()
                if not tokens:
                    continue
                cls = int(tokens[0])
                if cls in class_map:
                    tokens[0] = str(class_map[cls])
                new_lines.append(" ".join(tokens))

            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")
    print('Метки обработаны')

def preprocess_yolo_dataset():
    yaml_path = './car-parts-human-in-the-loop-1/data.yaml'
    preprocess_yaml(yaml_path)

    yolo_labels_path = [
        './car-parts-human-in-the-loop-1/train/labels',
        './car-parts-human-in-the-loop-1/valid/labels',
        './car-parts-human-in-the-loop-1/test/labels'
    ]
    for label_path in yolo_labels_path:
        preprocess_yolo_labels(label_path)

    print('Датасет yolo обработан')





if __name__ == '__main__':
    api_key = 'z4nj6C1okcmIqQQ6BCbg'
    dataset_yolo = load_dataset(
        api_key,
        'atheer-algarni-gvico',
        'car-parts-human-in-the-loop',
        'yolov8-obb'
        )
    print(f'Датасет загружен в: {dataset_yolo.location}')
    dataset_segm = load_dataset(
        api_key,
        'gp2-hknp7',
        'car-damage-detection-mwbgo',
        'coco-segmentation'
        )
    print(f'Датасет загружен в: {dataset_segm.location}')
    preprocess_damage_segmentation_dataset()
    preprocess_yolo_dataset()
