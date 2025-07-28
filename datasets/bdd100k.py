# datasets/bdd100k.py
import os
import torch
from PIL import Image
from torchvision.transforms import functional as F
import json

class BDD100KDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_dir=None, transforms=None, is_target=False):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.is_target = is_target

        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.is_target:
            target = {}  # No annotations used
        else:
            ann_path = os.path.join(self.ann_dir, img_name.replace(".jpg", ".jpg.json"))
            with open(ann_path, 'r') as f:
                ann = json.load(f)

            boxes = []
            labels = []
            for obj in ann.get('objects', []):
                if obj.get('geometryType') == "rectangle" and obj.get('points'):
                    pts = obj['points']['exterior']
                    if len(pts) == 2:
                        x1, y1 = pts[0]
                        x2, y2 = pts[1]
                        if x2 > x1 and y2 > y1:  # ensure valid box
                            boxes.append([x1, y1, x2, y2])
                            labels.append(self.label_to_id(obj['classTitle']))


            if len(boxes) == 0:
                # Skip image if no valid boxes
                return self.__getitem__((idx + 1) % len(self))

            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }

        if self.transforms:
            image = self.transforms(image)

        return image, target
    @staticmethod
    def label_to_id(category):
        class_map = {
            'car': 1, 'bus': 1, 'truck': 1, 'trailer': 1,
            'pedestrian': 2, 'rider': 2, 'person': 2,
            'bicycle': 3, 'motorcycle': 3,
        }
        return class_map.get(category, 0)
