"""
Training Orchestrator — runs all three training stages in sequence.

Usage
-----
# Full training (all three stages):
python run_training.py

# Quick test — 2 epochs, small subsets (verify the code runs before full training):
python run_training.py --quick-test

# Run only specific stages:
python run_training.py --stages yolo jersey scorer

# Skip a stage (e.g. if YOLO is already trained):
python run_training.py --stages jersey scorer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_config, get_device


VALID_STAGES = ["yolo", "jersey", "scorer"]


def run_stage_yolo(args: argparse.Namespace) -> None:
    from training.train_yolo import train, build_parser as yolo_parser
    sub = yolo_parser().parse_args([
        "--config", args.config,
        *(["--quick-test"] if args.quick_test else []),
        *(["--device", args.device] if args.device else []),
    ])
    train(sub)


def run_stage_jersey(args: argparse.Namespace) -> None:
    from training.train_jersey_cnn import train, build_parser as jersey_parser
    sub = jersey_parser().parse_args([
        "--config", args.config,
        *(["--quick-test"] if args.quick_test else []),
        *(["--device", args.device] if args.device else []),
    ])
    train(sub)


def run_stage_scorer(args: argparse.Namespace) -> None:
    from training.train_scorer_lstm import train, build_parser as scorer_parser
    sub = scorer_parser().parse_args([
        "--config", args.config,
        *(["--quick-test"] if args.quick_test else []),
        *(["--device", args.device] if args.device else []),
    ])
    train(sub)


STAGE_FNS = {
    "yolo":    run_stage_yolo,
    "jersey":  run_stage_jersey,
    "scorer":  run_stage_scorer,
}

STAGE_LABELS = {
    "yolo":   "Stage 1 — YOLOv8m Player Detection",
    "jersey": "Stage 2 — Two-Digit Jersey CNN",
    "scorer": "Stage 4 — ResNet-50 + LSTM Highlight Scorer",
}


def main(args: argparse.Namespace) -> None:
    stages: List[str] = args.stages or VALID_STAGES

    invalid = [s for s in stages if s not in VALID_STAGES]
    if invalid:
        print(f"[run_training] Unknown stage(s): {invalid}. Valid: {VALID_STAGES}")
        sys.exit(1)

    device = args.device or get_device()
    mode   = "QUICK-TEST" if args.quick_test else "FULL"

    print("=" * 65)
    print(f" Player Number-Guided AI Highlight Curator — Training")
    print(f" Mode   : {mode}")
    print(f" Device : {device}")
    print(f" Stages : {stages}")
    print("=" * 65)

    overall_start = time.time()

    for stage in stages:
        label = STAGE_LABELS[stage]
        print(f"\n{'─'*65}")
        print(f" Running: {label}")
        print(f"{'─'*65}")
        t0 = time.time()
        try:
            STAGE_FNS[stage](args)
        except Exception as exc:
            print(f"\n[run_training] ERROR in stage '{stage}': {exc}")
            if args.fail_fast:
                raise
            print("[run_training] Continuing to next stage (use --fail-fast to abort).")
        elapsed = time.time() - t0
        print(f"[run_training] '{stage}' finished in {elapsed:.1f}s")

    total = time.time() - overall_start
    print(f"\n{'='*65}")
    print(f" All requested stages complete in {total:.1f}s")
    print(f"{'='*65}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train all pipeline stages for the highlight curator."
    )
    p.add_argument(
        "--stages", nargs="+", default=None,
        choices=VALID_STAGES,
        help="Which stages to run (default: all three)",
    )
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--device",     default=None, help="cuda / mps / cpu")
    p.add_argument(
        "--quick-test", action="store_true",
        help="2-epoch quick test with small data subsets",
    )
    p.add_argument(
        "--fail-fast", action="store_true",
        help="Abort immediately if any stage raises an exception",
    )
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
