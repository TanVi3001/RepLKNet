from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flower_experiment import (  # noqa: E402
    DATASET_HANDLE,
    download_kaggle_flowers,
    run_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flowers Recognition: RepLKNet vs VGG-16")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "flowers")
    parser.add_argument("--download-kaggle", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "flowers")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if args.download_kaggle:
        print(f"Dataset: {DATASET_HANDLE}")
        data_dir = download_kaggle_flowers(data_dir)
        print(f"Đã chuẩn bị dữ liệu tại: {data_dir}")

    rows = run_experiment(
        data_dir=data_dir,
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        seeds=args.seeds,
        num_workers=args.num_workers,
        device_name=args.device,
        allow_random_init=args.allow_random_init,
        rebuild_split=args.rebuild_split,
        deterministic=not args.non_deterministic,
        resume=args.resume,
    )
    print(f"\nĐã lưu {len(rows)} kết quả model-seed vào: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
