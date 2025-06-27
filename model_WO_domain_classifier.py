# trainers/train_no_domain.py
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from datasets.bdd100k import BDD100KDataset
from models.da_raw_detector import DA_RAW_FasterRCNN
from losses.contrastive_loss import contrastive_loss
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


def train_one_epoch(model, src_loader, tgt_loader, optimizer, device,epoch):
    model.train()
    prototypes = {c: nn.Parameter(torch.randn(5, 128).to(device)) for c in range(6)}
    log_file = "logs/trainWO_domain_classifier.log"
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "a") as log:
        log.write(f"\n--- Epoch {epoch}: {datetime.now()} ---\n")

    for batch_idx, (src_batch, tgt_batch) in enumerate(zip(src_loader, tgt_loader)):
        src_imgs, src_targets = src_batch
        tgt_imgs, _ = tgt_batch
        src_imgs = [img.to(device) for img in src_imgs]
        tgt_imgs = [img.to(device) for img in tgt_imgs]
        src_targets = [{k: v.to(device) for k, v in t.items()} for t in src_targets]

        # Supervised source loss
        loss_dict = model(src_imgs, src_targets)
        loss_sup = sum(loss for loss in loss_dict.values())

        # Instance-level contrastive loss
        with torch.no_grad():
            model_mode = model.training
            model.eval()
            tgt_outputs = model.detector(tgt_imgs)
            batch_tensor = torch.stack(tgt_imgs)  # stack list into a single tensor of shape [B, C, H, W]
            features = model.detector.backbone(batch_tensor)
            instance_features = []
            pseudo_labels = []

            for i,out in enumerate(tgt_outputs):
                scores = out['scores']
                labels = out['labels']
                boxes = out['boxes']
                keep = scores > 0.8
                labels = labels[keep]
                boxes = boxes[keep]
                if len(boxes) == 0:
                    continue
                pooled_feat = model.detector.roi_heads.box_roi_pool(features,[boxes],tgt_imgs[i].shape[-2:])
                feat = model.project_instance_features(model.detector.roi_heads.box_head(pooled_feat))
                instance_features.append(feat)
                pseudo_labels.append(labels)
            model.train(model_mode)
            
        loss_contrastive = (
            contrastive_loss(prototypes, torch.cat(instance_features), torch.cat(pseudo_labels))
            if instance_features else torch.zeros(1, device=device).squeeze()
        )
        # Total loss (no domain loss)
        loss = loss_sup + loss_contrastive
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0:
            msg=f"Batch {batch_idx+1}: Loss={loss.item():.4f}, Sup={loss_sup.item():.4f}, Contrast={loss_contrastive.item():.4f}"
            print(msg)
            with open(log_file, "a") as log:
                log.write(msg + "\n")


def train_no_domain():
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
        train_one_epoch(model, src_loader, tgt_loader, optimizer, device,i+1)
        if (i + 1) % epochs == 0:
            ckpt_path = os.path.join(cfg['experiment']['output_dir'], f"da_without_domain_classifier_raw_epoch_{i+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    train_no_domain()
