# main.py
import numpy as np
import yaml
import torch
from trainers.train_da_raw import train_one_epoch
from models.da_raw_detector import DA_RAW_FasterRCNN
from datasets.bdd100k import BDD100KDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_dataloaders_from_config(cfg):
    tfms = transforms.Compose([
        transforms.Resize(tuple(cfg['data']['input_size'])),
        transforms.ToTensor()
    ])

    src_set = BDD100KDataset(
        img_dir=cfg['data']['source']['image_dir'],
        ann_dir=cfg['data']['source']['ann_dir'],
        transforms=tfms,
        is_target=False
    )

    tgt_set = BDD100KDataset(
        img_dir=cfg['data']['target']['image_dir'],
        transforms=tfms,
        is_target=True
    )

    src_loader = DataLoader(src_set, batch_size=cfg['data']['batch_size'], shuffle=True, collate_fn=lambda x: list(zip(*x)))
    tgt_loader = DataLoader(tgt_set, batch_size=cfg['data']['batch_size'], shuffle=True, collate_fn=lambda x: list(zip(*x)))
    return src_loader, tgt_loader


def compute_lambda(epoch, total_epochs=10):
    p = epoch / total_epochs
    return 2. / (1. + np.exp(-10 * p)) - 1

def main():
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DA_RAW_FasterRCNN(num_classes=cfg['model']['num_classes']).to(device)
    src_loader, tgt_loader = get_dataloaders_from_config(cfg)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg['optimizer']['lr'],
                                momentum=cfg['optimizer']['momentum'],
                                weight_decay=cfg['optimizer']['weight_decay'])

    os.makedirs(cfg['experiment']['output_dir'], exist_ok=True)
    epochs=cfg['experiment']['epochs']
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        model.grl.lambda_=compute_lambda(i+1,epochs)
        train_one_epoch(model, src_loader, tgt_loader, optimizer, device,i+1)
        if (i + 1) % epochs == 0:
            ckpt_path = os.path.join(cfg['experiment']['output_dir'], f"da_raw_epoch_{i+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()