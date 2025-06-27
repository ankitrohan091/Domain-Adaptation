# DA-RAW: Domain Adaptive Object Detection under Adverse Weather (BDD100K)

This repository implements the DA-RAW method for domain adaptive object detection under real-world weather (rain/snow) using the BDD100K dataset.

## 📁 Project Structure
```
.
├── configs/
│   └── config.yaml
├── datasets/
│   └── bdd100k.py
├── models/
│   ├── da_raw_detector.py
│   └── cbam.py
├── losses/
│   └── contrastive_loss.py
├── trainers/
│   └── train_da_raw.py
├── outputs/, checkpoints/  
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ Setup
```bash
pip install -r requirements.txt
```

## 📦 Dataset Setup
Dataset can be downloaded from "https://drive.google.com/file/d/1QPyVpJ8g4R1aUsXYvJdEwwzhPjIG6UT-/view?usp=sharing"
Organize BDD100K into:
```
datasets/
└── bdd100k/
    ├── source/
    │   ├── images/
    │   └── annotations/
    └── target/
        ├── images/
        └── annotations/ (optional for evaluation)
```
Each annotation must be in JSON format matching the image filename (e.g., `abc.jpg` → `abc.json`).

## 🚀 Training
```bash
python main.py
```
Checkpoints are saved to `checkpoints/` every epoch.

## 🧪 Evaluation
Use the provided script below to evaluate mAP on the target domain.

## 📜 Citation
Adapted from: *DA-RAW: Domain Adaptive Object Detection for Real-World Adverse Weather Conditions*, ICRA 2024
[📄 Download Paper (PDF)](https://arxiv.org/pdf/2309.08152)

## 📧 Contact
Ankit Kumar  - ankitrohan9113@gmail.com
For implementation support or research discussion, feel free to reach out.
