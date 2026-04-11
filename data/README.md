# Datasets

All three training stages use publicly available SoccerNet datasets.
Registration at **https://www.soccer-net.org/** is required (free academic access).

---

## Quick start (no credentials needed)

Generate tiny synthetic stubs to verify the training pipeline runs:

```bash
python scripts/download_soccernet.py --task stubs
python run_training.py --quick-test
```

---

## Full dataset download

Set your credentials once:

```bash
export SOCCERNET_USER="your@email.com"
export SOCCERNET_PASS="your_password"
```

Then download everything:

```bash
python scripts/download_soccernet.py --task all
```

Or download individual subsets:

```bash
python scripts/download_soccernet.py --task detection
python scripts/download_soccernet.py --task jersey
python scripts/download_soccernet.py --task actions
```

---

## Expected directory layout after download

```
data/
├── soccernet_detection/          ← Stage 1 (YOLOv8m)
│   ├── dataset.yaml
│   ├── images/
│   │   ├── train/  *.jpg
│   │   ├── val/    *.jpg
│   │   └── test/   *.jpg
│   └── labels/
│       ├── train/  *.txt   (YOLO format: class cx cy w h)
│       ├── val/    *.txt
│       └── test/   *.txt
│
├── soccernet_jersey/             ← Stage 2 (Jersey CNN)
│   ├── 00/  *.jpg               ← one folder per two-digit jersey number
│   ├── 01/  *.jpg
│   ├── ...
│   └── 99/  *.jpg
│
└── soccernet_actions/            ← Stage 4 (LSTM Scorer)
    └── clips/
        ├── clip_0000/
        │   ├── label.json        ← {"label": 0 or 1}
        │   ├── frame_0000.jpg
        │   └── ...
        └── clip_NNNN/
```

---

## Dataset citations

If you use these datasets in your work, please cite:

```
@inproceedings{Deliège2021SoccerNetv2,
  title     = {SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  author    = {Adrien Deliège et al.},
  booktitle = {CVPR Workshops},
  year      = {2021}
}

@inproceedings{Cioppa2022SoccerNetTracking,
  title     = {SoccerNet Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos},
  author    = {Anthony Cioppa et al.},
  booktitle = {CVPR Workshops},
  year      = {2022}
}
```
