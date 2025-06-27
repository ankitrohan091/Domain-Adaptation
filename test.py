# test.py
from datetime import datetime
import torch
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from models.da_raw_detector import DA_RAW_FasterRCNN

LABEL_MAP = {
    1: 'vehicle',
    2: 'person',
    3: 'bike',
}

def load_image(image_path):
    transform = T.Compose([
        T.Resize((800, 1280)),
        T.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image), image

def visualize_and_save(image_pil, boxes, scores, labels, threshold, save_path, img_name):
    plt.figure(figsize=(12, 8))
    plt.imshow(image_pil)
    ax = plt.gca()
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor='red', facecolor='none', linewidth=2))
            cls_name = LABEL_MAP.get(label.item(), str(label.item()))
            ax.text(x1, y1 - 5, f"{cls_name}: {score:.2f}", color='yellow', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5))
    plt.axis("off")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, img_name), bbox_inches='tight')
    plt.close()

def save_predictions_json(boxes, scores, labels, threshold, save_path, img_name):
    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            results.append({
                "bbox": [round(float(v), 2) for v in box],
                "score": round(float(score), 4),
                "label": int(label),
                "label_name": LABEL_MAP.get(int(label), str(label))
            })
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, img_name.replace(".jpg", ".json"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

def test(model_path, image_dir, threshold=0.8, output_dir="test_outputs"):
    log_file = "logs/test.log"
    with open(log_file, "a") as log:
        log.write(f"\n--- Test Run: {datetime.now()} ---\n")
        log.write(f"Model: {model_path}\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DA_RAW_FasterRCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()
    model.detector.roi_heads.nms_thresh = 0.3

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        img_tensor, img_pil = load_image(img_path)
        with torch.no_grad():
            output = model([img_tensor.to(device)])[0]
            boxes = output['boxes'].cpu()
            scores = output['scores'].cpu()
            labels = output['labels'].cpu()

        msg=f"Image: {img_name}, Detections: {len(boxes)}"
        print(msg)
        with open(log_file, "a")as log:
            log.write(f"{msg}\n")
        visualize_and_save(img_pil, boxes, scores, labels, threshold, os.path.join(output_dir, "images"), img_name)
        save_predictions_json(boxes, scores, labels, threshold, os.path.join(output_dir, "jsons"), img_name)

if __name__ == "__main__":
    test(
        model_path="checkpoints/da_raw_epoch_50.pth",
        image_dir="datasets/BDD100K/test/img",
        threshold=0.5,
        output_dir="test_outputs"
    )
