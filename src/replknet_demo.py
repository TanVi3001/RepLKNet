"""Các tiện ích nhỏ cho notebook demo RepLKNet.

File này nằm ngoài repository chính thức. Mục tiêu là giữ source gốc để đối chiếu,
đồng thời gom các phần lặp lại như load checkpoint, dự đoán, benchmark và ERF.
Các thư viện nặng chỉ được import bên trong hàm để notebook vẫn mở được khi máy
chưa cài PyTorch.
"""

from __future__ import annotations

import html
import importlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable


IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
REPLKNET_CHECKPOINTS = {
    "RepLKNet-31B_ImageNet-1K_224.pth": {
        "file_id": "1azQUiCxK9feYVkkrPqwVPBtNsTzDrX7S",
        "google_drive": (
            "https://drive.google.com/file/d/1azQUiCxK9feYVkkrPqwVPBtNsTzDrX7S/view"
        ),
        "resolution": "224x224",
        "pretraining": "ImageNet-1K",
    },
    "RepLKNet-31B_ImageNet-1K_384.pth": {
        "file_id": "1vo-P3XB6mRLUeDzmgv90dOu73uCeLfZN",
        "google_drive": (
            "https://drive.google.com/file/d/1vo-P3XB6mRLUeDzmgv90dOu73uCeLfZN/view"
        ),
        "resolution": "384x384",
        "pretraining": "ImageNet-1K",
    },
}


def find_project_root(start: str | Path | None = None) -> Path:
    """Tìm thư mục project dù notebook được mở từ root hay từ ``notebooks/``."""

    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "RepLKNet-pytorch" / "replknet.py").exists() and (
            candidate / "src" / "replknet_demo.py"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy project root. Hãy mở notebook trong project RepLKNet_Project."
    )


def device_report() -> dict[str, Any]:
    """Trả về thông tin Python/PyTorch/CUDA/GPU để in ở đầu notebook."""

    import platform
    import torch

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda or "Không có",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": "Không có GPU NVIDIA",
        "vram_gb": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu_name"] = props.name
        report["vram_gb"] = round(props.total_memory / (1024**3), 2)
    return report


def _official_module(project_root: str | Path):
    """Import module ``replknet.py`` từ repo chính thức mà không sửa source."""

    repo_path = str(Path(project_root).resolve() / "RepLKNet-pytorch")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return importlib.import_module("replknet")


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch cũ chưa có tham số weights_only.
        return torch.load(path, map_location="cpu")


def load_checkpoint_flexible(model, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load checkpoint dạng ``model``, ``state_dict`` hoặc state dict thuần.

    Một số checkpoint được lưu bởi DataParallel nên có tiền tố ``module.``;
    hàm này loại tiền tố đó và bỏ head nếu số lớp không khớp.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

    checkpoint = _torch_load(path)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_ema", "module"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Checkpoint không chứa state_dict dạng dictionary.")

    cleaned_state = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned_state[new_key] = value

    model_state = model.state_dict()
    removed_keys = []
    for key in ("head.weight", "head.bias"):
        if key in cleaned_state and key in model_state:
            if tuple(cleaned_state[key].shape) != tuple(model_state[key].shape):
                removed_keys.append(key)
                del cleaned_state[key]

    incompatible = model.load_state_dict(cleaned_state, strict=False)
    return {
        "path": str(path),
        "removed_keys": removed_keys,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "checkpoint_keys": len(cleaned_state),
    }


def download_official_checkpoint(
    destination: str | Path,
    checkpoint_name: str = "RepLKNet-31B_ImageNet-1K_224.pth",
) -> bool:
    """Tải checkpoint Google Drive chính thức nếu đã cài ``gdown``.

    Download không tự chạy trong notebook để tránh bất ngờ tải file vài trăm MB.
    Người dùng có thể bật tùy chọn trong notebook hoặc tải thủ công theo README.
    """

    if checkpoint_name not in REPLKNET_CHECKPOINTS:
        raise ValueError(
            f"Chưa có metadata checkpoint cho {checkpoint_name}. "
            f"Các lựa chọn: {sorted(REPLKNET_CHECKPOINTS)}"
        )
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Checkpoint đã có sẵn: {destination}")
        return True
    try:
        import gdown
    except ImportError:
        print("Chưa cài gdown; hãy chạy `python -m pip install gdown` hoặc tải thủ công.")
        return False

    file_id = REPLKNET_CHECKPOINTS[checkpoint_name]["file_id"]
    print(f"Đang tải checkpoint chính thức {checkpoint_name} ...")
    output = gdown.download(id=file_id, output=str(destination), quiet=False, fuzzy=True)
    ok = bool(output) and destination.exists() and destination.stat().st_size > 0
    print("Tải checkpoint thành công." if ok else "Tải checkpoint chưa thành công.")
    return ok


def build_replknet(
    project_root: str | Path,
    model_name: str = "RepLKNet-31B",
    checkpoint_path: str | Path | None = None,
    device=None,
    num_classes: int = 1000,
    merge_for_inference: bool = True,
):
    """Tạo RepLKNet từ implementation chính thức và load checkpoint nếu có."""

    import torch

    official = _official_module(project_root)
    factories = {
        "RepLKNet-31B": official.create_RepLKNet31B,
        "RepLKNet-31L": official.create_RepLKNet31L,
        "RepLKNet-XL": official.create_RepLKNetXL,
    }
    if model_name not in factories:
        raise ValueError(f"Model chưa được hỗ trợ: {model_name}")

    # use_checkpoint chỉ dành cho training; demo inference không cần giữ activation.
    model = factories[model_name](
        drop_path_rate=0.0,
        num_classes=num_classes,
        use_checkpoint=False,
        small_kernel_merged=False,
    )
    status: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_loaded": False,
        "checkpoint": None,
        "load_info": None,
        "merged_for_inference": False,
    }
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if checkpoint.exists():
            try:
                status["load_info"] = load_checkpoint_flexible(model, checkpoint)
                status["checkpoint_loaded"] = True
                status["checkpoint"] = str(checkpoint)
                print(f"Đã load checkpoint: {checkpoint.name}")
            except (RuntimeError, ValueError, OSError) as exc:
                print(f"Không load được checkpoint, tiếp tục với model chưa train: {exc}")
        else:
            print(
                "Chưa có checkpoint RepLKNet; demo vẫn chạy bằng trọng số khởi tạo "
                "nhưng dự đoán không có ý nghĩa ImageNet."
            )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if merge_for_inference:
        model.structural_reparam()
        status["merged_for_inference"] = True
    return model, status


def build_tiny_replknet_for_reparam_check(project_root: str | Path):
    """Tạo model nhỏ từ class chính thức để kiểm tra equivalence trước/sau merge."""

    official = _official_module(project_root)
    return official.RepLKNet(
        large_kernel_sizes=[7, 7],
        layers=[1, 1],
        channels=[8, 16],
        drop_path_rate=0.0,
        small_kernel=3,
        dw_ratio=1,
        ffn_ratio=2,
        in_channels=3,
        num_classes=10,
        use_checkpoint=False,
        small_kernel_merged=False,
        use_sync_bn=False,
    )


def get_image_transform(image_size: int = 224):
    from torchvision import transforms
    from PIL import Image

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=Image.Resampling.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def create_demo_image(path: str | Path, size: int = 512) -> Path:
    """Tạo ảnh hình học dễ nhìn nếu người dùng chưa bỏ ảnh thật vào ``images/``."""

    from PIL import Image, ImageDraw, ImageFont

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), (238, 243, 250))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 28, size - 28, size - 28), radius=28, fill=(35, 66, 99))
    draw.ellipse((92, 100, 292, 300), fill=(241, 180, 58), outline=(255, 245, 210), width=6)
    draw.rectangle((300, 104, 420, 300), fill=(76, 175, 112), outline=(225, 255, 230), width=6)
    draw.polygon([(120, 400), (230, 318), (340, 400)], fill=(211, 76, 76), outline=(255, 220, 220))
    draw.line((70, 348, 442, 348), fill=(255, 255, 255), width=5)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    draw.text((110, 54), "Demo image", fill=(255, 255, 255), font=font)
    image.save(path)
    return path


def find_or_create_sample_image(images_dir: str | Path) -> Path:
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in supported
    )
    if candidates:
        return candidates[0]
    return create_demo_image(images_dir / "demo_shapes.png")


def load_image_tensor(image_path: str | Path, image_size: int = 224):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tensor = get_image_transform(image_size)(image).unsqueeze(0)
    return image, tensor


def load_imagenet_labels(cache_path: str | Path | None = None) -> list[str]:
    """Đọc labels từ cache hoặc tải nguồn public; lỗi mạng thì dùng class id."""

    if cache_path is None:
        cache_path = Path("results") / "imagenet_classes.txt"
    cache_path = Path(cache_path)
    if not cache_path.exists():
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(IMAGENET_LABELS_URL, cache_path)
            print(f"Đã lưu nhãn ImageNet vào {cache_path}")
        except (OSError, urllib.error.URLError) as exc:
            print(f"Không tải được nhãn ImageNet ({exc}); dùng mã class thay thế.")
    if cache_path.exists():
        labels = [line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines()]
        if len(labels) >= 1000:
            return labels[:1000]
    return [f"class_{idx:04d}" for idx in range(1000)]


def predict_topk(model, image_tensor, labels: Iterable[str], device=None, k: int = 5):
    import torch

    if device is None:
        device = next(model.parameters()).device
    batch = image_tensor.to(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    values, indices = probabilities[0].topk(k)
    labels = list(labels)
    topk = [
        {
            "index": int(index),
            "label": labels[int(index)] if int(index) < len(labels) else f"class_{int(index):04d}",
            "confidence": float(value),
        }
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    return {"top1": topk[0], "top5": topk, "elapsed_ms": elapsed_ms}


def model_stats(model) -> dict[str, float | int]:
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    bytes_total = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    bytes_total += sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return {
        "parameters": int(params),
        "trainable_parameters": int(trainable),
        "model_size_mb": round(bytes_total / (1024**2), 2),
    }


def benchmark_model(model, image_tensor, warmup: int = 3, repeats: int = 10) -> dict[str, float | int | None]:
    """Benchmark có warm-up, nhiều lần lặp và synchronize GPU."""

    import torch

    device = next(model.parameters()).device
    batch = image_tensor.to(device)
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": round(sum(timings) / len(timings), 3),
        "std_ms": round((sum((t - sum(timings) / len(timings)) ** 2 for t in timings) / len(timings)) ** 0.5, 3),
        "warmup": warmup,
        "repeats": len(timings),
        "gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        if device.type == "cuda"
        else None,
    }


def _extract_feature(model, image_tensor, model_kind: str):
    if model_kind == "replknet":
        return model.forward_features(image_tensor)
    if model_kind == "vgg":
        # VGG-16 là CNN tuần tự; feature map cuối nằm trong ``features``.
        return model.features(image_tensor)
    raise ValueError("model_kind phải là 'replknet' hoặc 'vgg'.")


def compute_erf_map(model, image_tensor, model_kind: str = "replknet"):
    """Tính bản đồ gradient ERF đơn giản, cùng ý tưởng với script official ERF."""

    import numpy as np
    import torch

    device = next(model.parameters()).device
    samples = image_tensor.to(device).detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    features = _extract_feature(model, samples, model_kind)
    h, w = features.shape[-2:]
    central_score = torch.relu(features[:, :, h // 2, w // 2]).sum()
    gradient = torch.autograd.grad(central_score, samples, retain_graph=False)[0]
    contribution = torch.relu(gradient).sum(dim=(0, 1)).detach().cpu().numpy()
    contribution = np.log1p(contribution)
    max_value = float(contribution.max())
    if max_value > 0:
        contribution = contribution / max_value
    return contribution


def save_numpy(path: str | Path, array) -> Path:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path


def prediction_table_html(rows: list[dict[str, Any]]) -> str:
    """Tạo bảng HTML nhẹ, không bắt buộc cài pandas."""

    headers = ["Model", "Top-1", "Confidence", "Top-5", "Inference time (ms)"]
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        top5 = "<br>".join(
            f"{item['label']} ({item['confidence']:.2%})" for item in row["prediction"]["top5"]
        )
        body.append(
            "<tr>"
            f"<td>{html.escape(row['model'])}</td>"
            f"<td>{html.escape(row['prediction']['top1']['label'])}</td>"
            f"<td>{row['prediction']['top1']['confidence']:.2%}</td>"
            f"<td>{top5}</td>"
            f"<td>{row['prediction']['elapsed_ms']:.2f}</td>"
            "</tr>"
        )
    return (
        '<table style="border-collapse:collapse">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )
