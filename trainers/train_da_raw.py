# trainers/train_da_raw.py
import torch
import torch.nn as nn
from losses.contrastive_loss import contrastive_loss
import os
from datetime import datetime

def train_one_epoch(model, src_loader, tgt_loader, optimizer, device,epoch):
    model.train()
    prototypes = {c: nn.Parameter(torch.randn(5, 128).to(device)) for c in range(6)}
    log_file = "logs/train.log"
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "a") as log:
        log.write(f"\n--- Epoch {epoch}: {datetime.now()} ---\n")

    for batch_idx, (src_batch, tgt_batch) in enumerate(zip(src_loader, tgt_loader)):
        src_imgs, src_targets = src_batch
        tgt_imgs, _ = tgt_batch
        src_imgs = [img.to(device) for img in src_imgs]
        tgt_imgs = [img.to(device) for img in tgt_imgs]
        src_targets = [{k: v.to(device) for k, v in t.items()} for t in src_targets]

        loss_dict = model(src_imgs, src_targets)
        loss_sup = sum(loss for loss in loss_dict.values())

        fpn_feats = model.detector.backbone(tgt_imgs[0].unsqueeze(0))
        dom_logits_p4, dom_logits_p5 = model.image_level_alignment(fpn_feats)

        domain_labels = torch.ones_like(dom_logits_p4).to(device)
        loss_domain = nn.BCEWithLogitsLoss()(dom_logits_p4, domain_labels) + \
                      nn.BCEWithLogitsLoss()(dom_logits_p5, domain_labels)

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

        loss = loss_sup + loss_domain + loss_contrastive
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0:
            log_msg = (f"Batch {batch_idx+1}: Loss={loss.item():.4f}, Sup={loss_sup.item():.4f}, "
                       f"Domain={loss_domain.item():.4f}, Contrast={loss_contrastive.item():.4f}")
            print(log_msg)
            with open(log_file, "a") as log:
                log.write(log_msg + "\n")

