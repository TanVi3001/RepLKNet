"""Thực nghiệm phân loại Flowers Recognition cho RepLKNet và VGG-16.

Mã này giữ nguyên implementation RepLKNet chính thức và chỉ quản lý protocol
thực nghiệm bên ngoài repository đó. Hai model dùng cùng dữ liệu, split, phép
biến đổi ảnh, loss, optimizer, số epoch và seed.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_HANDLE = "alxmamaev/flowers-recognition"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_LABELS = ("VGG-16 (CNN thuần)", "RepLKNet-31B")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Đặt seed cho Python, NumPy và PyTorch để chạy lại cùng protocol."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _class_directories(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    class_dirs = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and any(
            file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
            for file in child.iterdir()
        ):
            class_dirs.append(child)
    return class_dirs


def locate_imagefolder_root(data_dir: str | Path) -> Path:
    """Tìm thư mục có cấu trúc ImageFolder trong thư mục Kaggle đã tải."""

    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

    candidates = [data_dir]
    candidates.extend(sorted(path for path in data_dir.iterdir() if path.is_dir()))
    for candidate in candidates:
        class_dirs = _class_directories(candidate)
        if len(class_dirs) >= 2:
            return candidate
    raise RuntimeError(
        "Không tìm thấy cấu trúc ImageFolder. Cần một thư mục chứa các thư mục lớp "
        "và bên trong là ảnh, ví dụ data/flowers/daisy/*.jpg."
    )


def download_kaggle_flowers(destination: str | Path) -> Path:
    """Tải dataset công khai từ KaggleHub và sao chép về thư mục project."""

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu kagglehub. Cài dependencies bằng `python -m pip install -r requirements.txt`."
        ) from exc

    cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()
    source_root = locate_imagefolder_root(cache_path)
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for class_dir in _class_directories(source_root):
        shutil.copytree(
            class_dir,
            destination / class_dir.name,
            dirs_exist_ok=True,
        )
    return locate_imagefolder_root(destination)


def build_transforms(image_size: int = 224):
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                int(round(image_size * 256 / 224)),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def stratified_split(
    targets: Iterable[int],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[int]]:
    """Chia index theo từng lớp để giữ phân bố lớp ở train/val/test."""

    targets = np.asarray(list(targets), dtype=np.int64)
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio và val_ratio phải dương, tổng nhỏ hơn 1.")

    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for class_index in sorted(np.unique(targets).tolist()):
        class_indices = np.flatnonzero(targets == class_index)
        rng.shuffle(class_indices)
        n_train = int(round(len(class_indices) * train_ratio))
        n_val = int(round(len(class_indices) * val_ratio))
        n_train = max(1, min(n_train, len(class_indices) - 2))
        n_val = max(1, min(n_val, len(class_indices) - n_train - 1))
        split["train"].extend(class_indices[:n_train].tolist())
        split["val"].extend(class_indices[n_train : n_train + n_val].tolist())
        split["test"].extend(class_indices[n_train + n_val :].tolist())

    for key in split:
        split[key].sort()
    return split


def _relative_samples(dataset, root: Path) -> list[str]:
    root = root.resolve()
    return [str(Path(path).resolve().relative_to(root)) for path, _ in dataset.samples]


def load_or_create_split(
    dataset,
    root: str | Path,
    manifest_path: str | Path,
    seed: int,
    force_rebuild: bool = False,
) -> dict[str, list[int]]:
    """Dùng lại split đã lưu; không âm thầm đổi test set giữa các lần chạy."""

    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    current_files = _relative_samples(dataset, root)
    if manifest_path.exists() and not force_rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("class_names") != dataset.classes or manifest.get("files") != current_files:
            raise RuntimeError(
                f"Split manifest không khớp dataset hiện tại: {manifest_path}. "
                "Dùng --rebuild-split nếu dataset đã thay đổi."
            )
        return {key: [int(index) for index in value] for key, value in manifest["indices"].items()}

    split = stratified_split(dataset.targets, seed=seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET_HANDLE,
        "dataset_root": str(root),
        "seed": seed,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "class_names": dataset.classes,
        "class_counts": {
            name: int(sum(target == index for target in dataset.targets))
            for index, name in enumerate(dataset.classes)
        },
        "files": current_files,
        "indices": split,
        "split_counts": {key: len(value) for key, value in split.items()},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return split


def make_loaders(
    root: Path,
    split: dict[str, list[int]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
):
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    train_transform, eval_transform = build_transforms(image_size)
    train_full = ImageFolder(root, transform=train_transform)
    eval_full = ImageFolder(root, transform=eval_transform)
    if train_full.samples != eval_full.samples or train_full.classes != eval_full.classes:
        raise RuntimeError("Train/eval ImageFolder không có cùng danh sách ảnh hoặc class mapping.")

    generator = __import__("torch").Generator().manual_seed(17_000 + seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(
            Subset(train_full, split["train"]),
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        ),
        "val": DataLoader(
            Subset(eval_full, split["val"]),
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            Subset(eval_full, split["test"]),
            shuffle=False,
            **loader_kwargs,
        ),
    }


def build_models(
    project_root: str | Path,
    num_classes: int,
    device,
    replk_checkpoint: str | Path | None,
    allow_random_init: bool = False,
):
    """Tạo hai model với ImageNet initialization và head 5 lớp."""

    import torch
    import torch.nn as nn
    from torchvision.models import VGG16_Weights, vgg16

    statuses: dict[str, dict[str, Any]] = {}
    try:
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)
        vgg_pretrained = True
    except Exception as exc:
        if not allow_random_init:
            raise RuntimeError(
                "Không tải được VGG-16 ImageNet weights. Dùng --allow-random-init "
                "chỉ khi muốn chạy kiểm tra pipeline, không dùng cho kết luận chính."
            ) from exc
        vgg_model = vgg16(weights=None)
        vgg_pretrained = False
        statuses["VGG-16 (CNN thuần)"] = {
            "pretrained": False,
            "pretrained_backbone": False,
            "classifier_reinitialized": True,
            "fallback_reason": str(exc),
        }
    vgg_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, num_classes)
    statuses.setdefault(
        "VGG-16 (CNN thuần)",
        {
            "pretrained": vgg_pretrained,
            "pretrained_backbone": vgg_pretrained,
            "classifier_reinitialized": True,
        },
    )

    if replk_checkpoint is None or not Path(replk_checkpoint).exists():
        if not allow_random_init:
            raise FileNotFoundError(
                "Không tìm thấy RepLKNet checkpoint. Đặt file vào weights/ hoặc dùng "
                "--allow-random-init cho smoke test."
            )
        checkpoint_for_model = None
    else:
        checkpoint_for_model = replk_checkpoint
    from replknet_demo import build_replknet

    replk_model, replk_status = build_replknet(
        project_root,
        model_name="RepLKNet-31B",
        checkpoint_path=checkpoint_for_model,
        # Giữ cả hai model trên CPU lúc khởi tạo; chỉ đưa model đang train lên
        # GPU để không chiếm VRAM không cần thiết của model còn lại.
        device=torch.device("cpu"),
        num_classes=num_classes,
        merge_for_inference=False,
    )
    statuses["RepLKNet-31B"] = {
        "pretrained": bool(replk_status["checkpoint_loaded"]),
        "pretrained_backbone": bool(replk_status["checkpoint_loaded"]),
        "classifier_reinitialized": True,
        "checkpoint": replk_status.get("checkpoint"),
        "load_info": replk_status.get("load_info"),
        "merged_for_inference": False,
    }
    if not replk_status["checkpoint_loaded"] and not allow_random_init:
        raise RuntimeError("RepLKNet checkpoint không được load; dừng để tránh kết luận sai.")

    return {
        "VGG-16 (CNN thuần)": vgg_model,
        "RepLKNet-31B": replk_model,
    }, statuses


def _classification_metrics(y_true, y_pred, class_names: list[str]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(
            precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )[2]
        ),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_model(model, loader, criterion, class_names: list[str], device) -> dict[str, Any]:
    import torch

    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            total_loss += float(loss.item()) * targets.size(0)
            total_items += targets.size(0)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    metrics = _classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / max(1, total_items)
    return metrics


def train_epoch(model, loader, criterion, optimizer, device, grad_clip: float) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    correct = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip > 0:
            __import__("torch").nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * targets.size(0)
        total_items += targets.size(0)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
    return {
        "loss": total_loss / max(1, total_items),
        "accuracy": correct / max(1, total_items),
    }


def _model_slug(model_label: str) -> str:
    return "vgg16" if model_label.startswith("VGG") else "replknet31b"


def measure_latency(model, loader, device, warmup: int = 10, repeats: int = 30) -> dict[str, Any]:
    import torch

    images, _ = next(iter(loader))
    batch = images[:1].to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    mean_ms = float(np.mean(timings))
    return {
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(float(np.std(timings)), 3),
        "gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        if device.type == "cuda"
        else None,
    }


def save_confusion_matrix(metrics: dict[str, Any], class_names: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_history_plot(history: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="val macro-F1")
    axes[1].set_title("Accuracy / macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_comparison_plot(summary_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    model_names = list(MODEL_LABELS)
    metric_names = ["test_accuracy", "test_macro_f1", "test_balanced_accuracy"]
    metric_titles = ["Accuracy", "Macro-F1", "Balanced accuracy"]
    means = []
    errors = []
    for metric in metric_names:
        means.append(
            [
                float(np.mean([row[metric] for row in summary_rows if row["model"] == model]))
                for model in model_names
            ]
        )
        errors.append(
            [
                float(np.std([row[metric] for row in summary_rows if row["model"] == model], ddof=1))
                if len([row for row in summary_rows if row["model"] == model]) > 1
                else 0.0
                for model in model_names
            ]
        )
    x = np.arange(len(model_names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, title, values, stds in zip(axes, metric_titles, means, errors):
        axis.bar(x, values, yerr=stds, capsize=5, color=["#4C78A8", "#F58518"])
        axis.set_title(title)
        axis.set_xticks(x, ["VGG-16", "RepLKNet-31B"], rotation=20)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Flowers Recognition — mean ± standard deviation across seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_conclusion(summary_rows: list[dict[str, Any]], output_dir: Path, seeds: list[int]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(row["model"], []).append(row)
    means = {
        model: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("test_accuracy", "test_macro_f1", "test_balanced_accuracy")
        }
        for model, rows in grouped.items()
    }
    best_model = max(means, key=lambda model: means[model]["test_macro_f1"])
    config = json.loads((output_dir / "experiment_config.json").read_text(encoding="utf-8"))
    lines = [
        "# Kết luận thực nghiệm Flowers Recognition",
        "",
        f"Protocol đã chạy với {len(seeds)} seed: {', '.join(str(seed) for seed in seeds)}.",
        f"Mỗi run dùng {config['epochs']} epoch, batch size {config['batch_size']} và "
        f"learning rate {config['learning_rate']}.",
        "Hai model dùng cùng split stratified 70/15/15, cùng preprocessing 224×224, "
        "loss CrossEntropy có label smoothing, AdamW và lịch CosineAnnealingLR.",
        "",
        "## Kết quả trung bình theo seed",
        "",
        "| Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) | Balanced accuracy (mean ± std) |",
        "|---|---:|---:|---:|",
    ]
    for model, values in means.items():
        rows = grouped[model]
        stds = {
            metric: float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else 0.0
            for metric in ("test_accuracy", "test_macro_f1", "test_balanced_accuracy")
        }
        lines.append(
            f"| {model} | {values['test_accuracy']:.4f} ± {stds['test_accuracy']:.4f} | "
            f"{values['test_macro_f1']:.4f} ± {stds['test_macro_f1']:.4f} | "
            f"{values['test_balanced_accuracy']:.4f} ± {stds['test_balanced_accuracy']:.4f} |"
        )
    lines.append("")
    for model in MODEL_LABELS:
        rows = grouped[model]
        macro_f1_std = float(np.std([row["test_macro_f1"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0
        ci95 = 1.96 * macro_f1_std / np.sqrt(len(rows)) if len(rows) > 1 else None
        ci_text = f"± {ci95:.4f}" if ci95 is not None else "không ước lượng được với một seed"
        lines.append(f"95% CI xấp xỉ của macro-F1 {model}: {ci_text}.")
    if config["epochs"] < 10:
        lines.extend(
            [
                "",
                "Run artifact hiện tại dùng 3 epoch để kiểm chứng pipeline và so sánh sample-efficiency; "
                "kết luận cuối cho báo cáo nên dùng run dài hơn sau khi kiểm tra đường cong hội tụ.",
            ]
        )
    paired_deltas = []
    for seed in seeds:
        seed_rows = {row["model"]: row for row in summary_rows if row["seed"] == seed}
        if all(model in seed_rows for model in MODEL_LABELS):
            paired_deltas.append(
                {
                    "seed": seed,
                    "accuracy": seed_rows["RepLKNet-31B"]["test_accuracy"]
                    - seed_rows["VGG-16 (CNN thuần)"]["test_accuracy"],
                    "macro_f1": seed_rows["RepLKNet-31B"]["test_macro_f1"]
                    - seed_rows["VGG-16 (CNN thuần)"]["test_macro_f1"],
                }
            )
    if paired_deltas:
        delta_f1 = np.asarray([row["macro_f1"] for row in paired_deltas])
        delta_acc = np.asarray([row["accuracy"] for row in paired_deltas])
        lines.extend(
            [
                "",
                "## Chênh lệch ghép cặp theo seed",
                "",
                f"RepLKNet-31B − VGG-16: accuracy trung bình {delta_acc.mean():+.4f}, "
                f"macro-F1 trung bình {delta_f1.mean():+.4f} "
                f"(độ lệch chuẩn macro-F1 {delta_f1.std(ddof=1) if len(delta_f1) > 1 else 0.0:.4f}).",
            ]
        )
    lines.extend(
        [
            "",
            f"Theo macro-F1 trung bình trên test split hiện tại, model đạt kết quả cao hơn là **{best_model}**.",
            "Kết luận này chỉ có giá trị trong phạm vi dataset, split, protocol và phần cứng đã ghi trong artifacts.",
            "Không dùng latency hoặc một ảnh demo để thay thế đánh giá trên test set.",
            "VGG-16 có nhiều tham số hơn RepLKNet-31B; chênh lệch accuracy không được quy trực tiếp "
            "cho large kernel nếu chưa kiểm soát capacity bằng một thí nghiệm bổ sung.",
        ]
    )
    (output_dir / "conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    data_dir: str | Path,
    project_root: str | Path,
    output_dir: str | Path = "results/flowers",
    replk_checkpoint: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    image_size: int = 224,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    seeds: Iterable[int] = (42,),
    num_workers: int = 0,
    device_name: str | None = None,
    allow_random_init: bool = False,
    rebuild_split: bool = False,
    deterministic: bool = True,
    resume: bool = False,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn as nn
    from torchvision.datasets import ImageFolder

    project_root = Path(project_root).resolve()
    root = locate_imagefolder_root(data_dir)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("Cần ít nhất một seed.")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, eval_transform = build_transforms(image_size)
    index_dataset = ImageFolder(root, transform=eval_transform)
    if len(index_dataset.classes) != 5:
        raise RuntimeError(
            f"Dataset đang có {len(index_dataset.classes)} lớp ({index_dataset.classes}), "
            "không khớp Flowers Recognition 5 lớp."
        )

    split_path = output_dir / "split_manifest.json"
    split = load_or_create_split(
        index_dataset,
        root=root,
        manifest_path=split_path,
        seed=seeds[0],
        force_rebuild=rebuild_split,
    )
    protocol = {
        "dataset": DATASET_HANDLE,
        "data_root": str(root),
        "class_names": index_dataset.classes,
        "image_size": image_size,
        "split": {key: len(value) for key, value in split.items()},
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "grad_clip": grad_clip,
        "seeds": seeds,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "deterministic": deterministic,
        "pretrained_initialization": True,
    }
    previous_config_path = output_dir / "experiment_config.json"
    previous_config = (
        json.loads(previous_config_path.read_text(encoding="utf-8"))
        if resume and previous_config_path.exists()
        else None
    )
    _write_json(protocol, previous_config_path)

    if previous_config is not None:
        # Cho phép bổ sung seed khi resume, nhưng không trộn artifacts từ protocol khác.
        protocol_keys = (
            "data_root",
            "class_names",
            "image_size",
            "split",
            "epochs",
            "batch_size",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "label_smoothing",
            "grad_clip",
            "device",
            "torch",
            "cuda",
            "gpu",
            "deterministic",
            "pretrained_initialization",
        )
        for key in protocol_keys:
            if previous_config.get(key) != protocol.get(key):
                raise RuntimeError(
                    f"Protocol hiện tại khác experiment_config.json ở key '{key}'. "
                    "Bỏ --resume hoặc dùng đúng hyperparameters của run cũ."
                )

    all_summary_rows: list[dict[str, Any]] = []
    if resume and (output_dir / "summary.json").exists():
        all_summary_rows.extend(json.loads((output_dir / "summary.json").read_text(encoding="utf-8")))
    checkpoint_path = replk_checkpoint
    if checkpoint_path is None:
        checkpoint_path = project_root / "weights" / "RepLKNet-31B_ImageNet-1K_224.pth"

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        if resume:
            existing_rows = []
            for model_label, slug in zip(MODEL_LABELS, ("vgg16", "replknet31b")):
                metrics_path = seed_dir / f"{slug}_metrics.json"
                if not metrics_path.exists():
                    existing_rows = []
                    break
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
                result.setdefault("pretrained_backbone", result.get("pretrained", False))
                result.setdefault("classifier_reinitialized", True)
                existing_rows.append(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in {"per_class", "confusion_matrix"}
                    }
                )
            if len(existing_rows) == len(MODEL_LABELS):
                all_summary_rows = [row for row in all_summary_rows if row.get("seed") != seed]
                all_summary_rows.extend(existing_rows)
                print(f"seed={seed}: đã có đủ artifacts, bỏ qua khi --resume")
                continue
        set_seed(seed, deterministic=deterministic)
        seed_dir.mkdir(parents=True, exist_ok=True)
        loaders = make_loaders(
            root,
            split,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
        )
        models, statuses = build_models(
            project_root,
            num_classes=len(index_dataset.classes),
            device=device,
            replk_checkpoint=checkpoint_path,
            allow_random_init=allow_random_init,
        )
        for model_label in MODEL_LABELS:
            set_seed(seed, deterministic=deterministic)
            model = models[model_label].to(device)
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            history: list[dict[str, Any]] = []
            best_score = -float("inf")
            best_epoch = 0
            best_state = None
            started = time.perf_counter()
            print(f"\n[{model_label}] seed={seed} device={device}")
            for epoch in range(1, epochs + 1):
                epoch_started = time.perf_counter()
                train_metrics = train_epoch(model, loaders["train"], criterion, optimizer, device, grad_clip)
                val_metrics = evaluate_model(model, loaders["val"], criterion, index_dataset.classes, device)
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
                history.append(row)
                if val_metrics["macro_f1"] > best_score:
                    best_score = val_metrics["macro_f1"]
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                scheduler.step()
                print(
                    f"epoch {epoch:02d}/{epochs} | train loss {row['train_loss']:.4f} "
                    f"| val acc {row['val_accuracy']:.4f} | val macro-F1 {row['val_macro_f1']:.4f}"
                )

            training_seconds = time.perf_counter() - started
            if best_state is None:
                raise RuntimeError(f"Không có checkpoint tốt nhất cho {model_label}.")
            model.load_state_dict(best_state)
            model.eval()
            if model_label == "RepLKNet-31B":
                model.structural_reparam()
                statuses[model_label]["merged_for_inference"] = True
            test_metrics = evaluate_model(model, loaders["test"], criterion, index_dataset.classes, device)
            latency = measure_latency(model, loaders["test"], device)
            from replknet_demo import model_stats

            stats = model_stats(model)
            slug = _model_slug(model_label)
            checkpoint_file = seed_dir / f"{slug}_best.pt"
            torch.save(
                {
                    "model": model_label,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "class_names": index_dataset.classes,
                    "state_dict": best_state,
                    "protocol": protocol,
                },
                checkpoint_file,
            )
            _write_json(history, seed_dir / f"{slug}_history.json")
            save_history_plot(history, seed_dir / f"{slug}_history.png")
            save_confusion_matrix(
                test_metrics,
                index_dataset.classes,
                seed_dir / f"{slug}_confusion_matrix.png",
            )
            result = {
                "seed": seed,
                "model": model_label,
                "best_epoch": best_epoch,
                "training_seconds": round(training_seconds, 3),
                "pretrained": statuses[model_label]["pretrained"],
                "pretrained_backbone": statuses[model_label]["pretrained_backbone"],
                "classifier_reinitialized": statuses[model_label]["classifier_reinitialized"],
                "merged_for_inference": statuses[model_label].get("merged_for_inference", False),
                "parameters": stats["parameters"],
                "model_size_mb": stats["model_size_mb"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_precision": test_metrics["macro_precision"],
                "test_macro_recall": test_metrics["macro_recall"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_loss": test_metrics["loss"],
                "latency_mean_ms": latency["mean_ms"],
                "latency_std_ms": latency["std_ms"],
                "gpu_memory_mb": latency["gpu_memory_mb"],
                "per_class": test_metrics["per_class"],
                "confusion_matrix": test_metrics["confusion_matrix"],
                "checkpoint": str(checkpoint_file),
            }
            _write_json(result, seed_dir / f"{slug}_metrics.json")
            all_summary_rows.append(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"per_class", "confusion_matrix"}
                }
            )
            print(
                f"[{model_label}] test accuracy={result['test_accuracy']:.4f}, "
                f"macro-F1={result['test_macro_f1']:.4f}, latency={result['latency_mean_ms']:.3f} ms"
            )
            models[model_label] = model.to("cpu")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_json(all_summary_rows, output_dir / "summary.json")
    _write_csv(all_summary_rows, output_dir / "summary.csv")
    aggregates = {}
    for model_label in MODEL_LABELS:
        rows = [row for row in all_summary_rows if row["model"] == model_label]
        aggregates[model_label] = {
            metric: {
                "n": len(rows),
                "mean": float(np.mean([row[metric] for row in rows])),
                "std": float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "ci95": float(
                    1.96 * np.std([row[metric] for row in rows], ddof=1) / np.sqrt(len(rows))
                )
                if len(rows) > 1
                else None,
            }
            for metric in (
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_precision",
                "test_macro_recall",
                "test_macro_f1",
                "test_weighted_f1",
                "latency_mean_ms",
            )
        }
    _write_json(aggregates, output_dir / "aggregate_metrics.json")
    save_comparison_plot(all_summary_rows, output_dir / "comparison_metrics.png")
    _write_conclusion(all_summary_rows, output_dir, seeds)
    return all_summary_rows
