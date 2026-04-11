# Player Number-Guided AI Highlight Video Curator

**Mateusz Muszynski and Colin Wallace**  
Course CSCI 5922 — University of Colorado Boulder

---

## Abstract

A player number-guided AI system for automatically curating highlight videos from sports footage. Given a target jersey number, the four-stage pipeline detects players with **YOLOv8m**, reads jersey numbers via a **two-step CNN classifier**, tracks players with **ByteTrack + jersey-number identity anchoring**, and scores/assembles highlights with a **ResNet-50 + LSTM** scorer.

---

## Pipeline Overview

```
Raw Video + Jersey Number
        │
        ▼
┌─────────────────────┐
│  Stage 1: Detect    │  YOLOv8m — bounding box per player per frame
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 2: Read #    │  Orientation filter → 2-digit CNN classifier
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 3: Track     │  ByteTrack + jersey-number hard identity anchor
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 4: Score &   │  ResNet-50 + LSTM → clip relevance score → FFmpeg
│  Assemble           │
└─────────────────────┘
          │
          ▼
    Highlight Reel (.mp4)
```

---

## Repository Structure

```
CSCI 5922 - Final Project/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config.yaml              ← all hyperparameters & paths
├── main.py                  ← run the full pipeline (inference)
├── run_training.py          ← orchestrate all three training stages
├── src/
│   ├── detector.py          ← Stage 1: YOLOv8m player detection
│   ├── jersey_reader.py     ← Stage 2: CNN jersey number recognition
│   ├── tracker.py           ← Stage 3: ByteTrack + jersey identity anchor
│   ├── scorer.py            ← Stage 4: ResNet-50 + LSTM highlight scoring
│   └── utils.py             ← shared helpers (video I/O, visualization)
├── training/
│   ├── train_yolo.py        ← fine-tune YOLOv8m on SoccerNet detection
│   ├── train_jersey_cnn.py  ← train two-digit jersey CNN
│   └── train_scorer_lstm.py ← train ResNet-50 + LSTM scorer
├── notebooks/
│   └── 01_experiments.ipynb ← reproduce Experiments 1 & 2
├── data/
│   └── README.md            ← dataset download instructions
├── models/                  ← saved model weights (git-ignored)
├── outputs/                 ← highlight reels (git-ignored)
├── logs/                    ← TensorBoard / CSV logs (git-ignored)
└── scripts/
    └── download_soccernet.py ← automated SoccerNet data download
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-org>/csci5922-highlight-curator.git
cd "csci5922-highlight-curator"
pip install -r requirements.txt
```

> **GPU strongly recommended.** Tested on Python 3.10+, PyTorch 2.x, CUDA 11.8+.

### 2. Download datasets

```bash
python scripts/download_soccernet.py --task all
```

See `data/README.md` for manual download instructions and directory layout.

### 3. Train all stages

```bash
# Full training
python run_training.py

# Quick test (small subset, 2 epochs) — verify everything runs
python run_training.py --quick-test
```

Or train individual stages:

```bash
python training/train_yolo.py
python training/train_jersey_cnn.py
python training/train_scorer_lstm.py
```

### 4. Run the full pipeline on a video

```bash
python main.py --video path/to/match.mp4 --jersey 10 --output outputs/player10_highlights.mp4
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--jersey` | required | Target jersey number (0–99) |
| `--conf` | 0.3 | YOLO detection confidence threshold |
| `--highlight-thresh` | 0.6 | LSTM highlight score threshold |
| `--device` | auto | `cpu`, `cuda`, or `mps` |
| `--config` | config.yaml | Path to config file |

---

## Configuration

All hyperparameters live in `config.yaml`. Key sections:

- `detector` — YOLO confidence / IOU thresholds  
- `jersey_reader` — CNN confidence, orientation threshold  
- `tracker` — ByteTrack buffer frames, jersey reid weight  
- `scorer` — LSTM hidden dim, clip length, highlight threshold  
- `training.*` — epochs, batch sizes, quick-test overrides  

---

## Experiments

See `notebooks/01_experiments.ipynb` for:

- **Experiment 1** — Jersey number recognition accuracy (top-1 digit-pair, per-digit, orientation filter recall)  
- **Experiment 2** — Full pipeline highlight retrieval (precision / recall / F1 at clip level, identity accuracy)

---

## Datasets

| Dataset | Use | License |
|---------|-----|---------|
| [SoccerNet Player Detection](https://www.soccer-net.org/) | Train Stage 1 (YOLO) | SoccerNet Research |
| [SoccerNet Jersey Number Recognition](https://www.soccer-net.org/) | Train Stage 2 (CNN) | SoccerNet Research |
| [SoccerNet Action Spotting](https://www.soccer-net.org/) | Train Stage 4 (LSTM) | SoccerNet Research |

Registration at [soccer-net.org](https://www.soccer-net.org/) is required to download data.

---

## References

1. Rangasamy et al. — *Deep learning in sport video analysis: a review.* TELKOMNIKA 18(4), 2020.  
2. Gao et al. — *Sports video classification method based on improved deep learning.* Applied Sciences 14(2), 2024.  
3. Zawbaa et al. — *Event detection based approach for soccer video summarization.* IJMUE 7(2), 2012.  
4. Cust et al. — *Machine and deep learning for sport-specific movement recognition: a systematic review.* J. Sports Sciences 37(5), 2019.

---

## License

MIT — see `LICENSE`.
