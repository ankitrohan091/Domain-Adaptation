# evaluate.py
import torch
from torchvision.ops import box_iou
from models.da_raw_detector import DA_RAW_FasterRCNN
from datasets.bdd100k import BDD100KDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from datetime import datetime


def calculate_map(preds, gts, iou_thresh=0.5):
    tp, fp, fn = 0, 0, 0
    for pred, gt in zip(preds, gts):
        if pred.numel() == 0 and gt.numel() == 0:
            continue
        elif pred.numel() == 0:
            fn += gt.size(0)
            continue
        elif gt.numel() == 0:
            fp += pred.size(0)
            continue

        # ✅ Shape check and reshape if needed
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if gt.dim() == 1:
            gt = gt.unsqueeze(0)

        if pred.size(1) != 4 or gt.size(1) != 4:
            print(f"[Warning] Skipping invalid boxes: pred shape={pred.shape}, gt shape={gt.shape}")
            continue

        ious = box_iou(pred, gt)
        matches = (ious > iou_thresh).sum().item()
        tp += matches
        fp += pred.size(0) - matches
        fn += gt.size(0) - matches

    precision = tp / (tp + fp + 1e-8)  #true positive/prediction
    recall = tp / (tp + fn + 1e-8)     #tp/(true positive+not matched ground truth)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def compute_ap(preds, gts, iou_thresh=0.5):
    scores = []
    total_gt = 0

    for pred, gt in zip(preds, gts):
        total_gt += len(gt)
        if len(pred) == 0:
            continue
        ious = box_iou(pred, gt) if len(gt) > 0 else torch.empty((len(pred), 0))
        matched = torch.zeros(len(gt), dtype=torch.bool)

        for i in range(len(pred)):
            max_iou, max_idx = 0.0, -1
            for j in range(len(gt)):
                if matched[j]:
                    continue
                iou = ious[i, j]
                if iou >= iou_thresh and iou > max_iou:
                    max_iou = iou.item()
                    max_idx = j
            if max_idx >= 0:
                scores.append(1)
                matched[max_idx] = True
            else:
                scores.append(0)

    if not scores or total_gt == 0:
        return 0.0

    scores = torch.tensor(scores, dtype=torch.float32)
    tps = torch.cumsum(scores, dim=0)
    fps = torch.cumsum(1 - scores, dim=0)
    precisions = tps / (tps + fps + 1e-8)
    recalls = tps / (total_gt + 1e-8)

    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        if (recalls >= t).any():
            ap += precisions[recalls >= t].max()
    ap /= 11
    return ap.item()


def evaluate(model_path, data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DA_RAW_FasterRCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((800, 1280)),
        transforms.ToTensor()
    ])
    dataset = BDD100KDataset(
        img_dir=os.path.join(data_root, "img"),
        ann_dir=os.path.join(data_root, "ann"),
        transforms=tfms,
        is_target=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    preds, gts = [], []
    pred_by_class = {c: [] for c in range(1, 4)}
    gt_by_class = {c: [] for c in range(1, 4)}

    os.makedirs("logs", exist_ok=True)
    # log_file = "logs/eval.log"
    log_file = "logs/evalWO_domain_classifier.log"

    with open(log_file, "a") as log:
        log.write(f"\n--- Evaluation Run: {datetime.now()} ---\n")
        log.write(f"Model: {model_path}\n")

    with torch.no_grad():
        for i, (img, target) in enumerate(tqdm(loader)):
            img = img[0].to(device)
            out = model([img])[0]
            scores = out['scores']
            labels = out['labels']
            boxes = out['boxes']

            keep = scores >= 0.5
            boxes = boxes[keep].cpu()
            labels = labels[keep].cpu()

            gt_boxes = target['boxes'].squeeze(0)
            gt_labels = target['labels'].squeeze(0)

            preds.append(boxes)
            gts.append(gt_boxes)

            for c in range(1, 4):
                pred_c = boxes[labels == c]
                gt_c = gt_boxes[gt_labels == c]
                pred_by_class[c].append(pred_c)
                gt_by_class[c].append(gt_c)

            if (i + 1) % 100 == 0:
                with open(log_file, "a") as log:
                    log.write(f"Processed {i+1} images\n")

    # Precision / Recall / F1
    p, r, f1 = calculate_map(preds, gts)

    # mAP@0.5
    aps_50 = [compute_ap(pred_by_class[c], gt_by_class[c], iou_thresh=0.5) for c in range(1, 4)]
    mAP_50 = sum(aps_50) / len(aps_50)

    # mAP@[.5:.95]
    aps_all = []
    for iou_thresh in torch.arange(0.5, 0.96, 0.05):
        for c in range(1, 4):
            ap = compute_ap(pred_by_class[c], gt_by_class[c], iou_thresh=iou_thresh.item())
            aps_all.append(ap)
    mAP_all = sum(aps_all) / len(aps_all)

    # Log results
    with open(log_file, "a") as log:
        log.write(f"Final Precision: {p:.4f}\n")
        log.write(f"Final Recall: {r:.4f}\n")
        log.write(f"Final F1 Score: {f1:.4f}\n")
        for c, ap in enumerate(aps_50, 1):
            log.write(f"Class {c}: AP@0.5 = {ap:.4f}\n")
        log.write(f"mAP@0.5: {mAP_50:.4f}\n")
        log.write(f"mAP@[.5:.95]: {mAP_all:.4f}\n")

    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f1:.4f}, mAP@0.5: {mAP_50:.4f}, mAP@[.5:.95]: {mAP_all:.4f}")


if __name__ == '__main__':
    # evaluate(model_path='checkpoints/da_raw_epoch_50.pth', data_root='datasets/BDD100K/val')

    #Evaluate Model without domain classifier
    evaluate(model_path='checkpoints/da_without_domain_classifier_raw_epoch_50.pth', data_root='datasets/BDD100K/val')
