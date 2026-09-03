# RepLKNet (2022) – Large Kernel CNN và CNN truyền thống

Project demo phục vụ thuyết trình trên lớp. Nội dung trả lời ba câu hỏi WHAT –
WHY – HOW, sau đó chạy inference, benchmark và minh họa Effective Receptive
Field (ERF) của RepLKNet so với VGG-16, một CNN thuần truyền thống.

RepLKNet vẫn là một CNN. Điểm khác là RepLKNet dùng các depthwise convolution
kernel lớn, nổi bật với 31×31, còn CNN truyền thống thường dựa nhiều vào kernel
3×3. Implementation được dùng là repository PyTorch chính thức của tác giả:
[DingXiaoH/RepLKNet-pytorch](https://github.com/DingXiaoH/RepLKNet-pytorch).

## Cấu trúc

```text
RepLKNet/
├── RepLKNet-pytorch/       # Repository chính thức, giữ nguyên để đối chiếu
├── notebooks/
│   └── RepLKNet_Demo_Vietnamese.ipynb
├── src/
│   └── replknet_demo.py    # Wrapper/helper bên ngoài source chính thức
├── images/                 # Ảnh đầu vào; notebook tự tạo ảnh mẫu nếu trống
├── weights/                # Đặt checkpoint .pth tại đây
├── results/
│   ├── predictions/
│   ├── comparison/
│   └── erf/
├── requirements.txt
└── README.md
```

## Môi trường và cài đặt

Khuyến nghị Python 3.10–3.12 và PyTorch có CUDA phù hợp với GPU NVIDIA local.
Không cần sửa source trong `RepLKNet-pytorch/`.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate, có thể chạy trực tiếp bằng
`.venv\Scripts\python.exe` và `.venv\Scripts\jupyter.exe`.

### Cài PyTorch có CUDA cho GPU NVIDIA

`requirements.txt` giữ ràng buộc PyTorch ở mức version để không khóa một CUDA
toolkit cụ thể. Với GPU NVIDIA, hãy dùng [PyTorch Start Locally](https://pytorch.org/get-started/locally/)
để chọn đúng command theo Windows, phiên bản Python và CUDA; chạy command đó
trước `pip install -r requirements.txt`. Sau đó kiểm tra:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu kết quả có hậu tố `+cpu` hoặc `False`, PyTorch hiện tại là CPU build dù máy
có GPU; hãy cài lại wheel CUDA theo selector chính thức. Notebook vẫn tự chuyển
sang CPU nên không bị lỗi, nhưng benchmark large-kernel sẽ chậm hơn đáng kể.

Trên máy RTX 3050 6GB của project này, cặp wheel đã kiểm thử thành công là:

```powershell
python -m pip install --upgrade --force-reinstall torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Không cần cài `nvcc` riêng để chạy notebook; wheel PyTorch đã mang theo runtime
CUDA cần thiết. Nếu PyTorch phát hành phiên bản mới, ưu tiên command do selector
chính thức cung cấp.

## Checkpoint cần tải

Notebook mặc định dùng **RepLKNet-31B, ImageNet-1K, input 224×224**. Checkpoint
chính thức là `RepLKNet-31B_ImageNet-1K_224.pth`, khoảng vài trăm MB, lấy từ
link Google Drive trong README của repository chính thức. Có hai cách:

```powershell
python -m gdown 1azQUiCxK9feYVkkrPqwVPBtNsTzDrX7S -O weights/RepLKNet-31B_ImageNet-1K_224.pth
```

Hoặc tải thủ công từ [Google Drive chính thức](https://drive.google.com/file/d/1azQUiCxK9feYVkkrPqwVPBtNsTzDrX7S/view), rồi đặt đúng tên file vào `weights/`.

Notebook có sẵn hàm download nhưng mặc định không tự tải file lớn. Chỉ cần đổi
`AUTO_DOWNLOAD_CHECKPOINT = True` trong phần 6 nếu muốn notebook thực hiện việc
này. Nếu chưa có checkpoint, notebook vẫn mở và chạy được các phần lý thuyết,
minh họa kernel, kiểm tra structural re-parameterization bằng model nhỏ; phần
dự đoán RepLKNet sẽ ghi rõ đang dùng trọng số khởi tạo nên không có ý nghĩa
ImageNet.

VGG-16 ImageNet pretrained được tải tự động bởi `torchvision` ở lần chạy đầu.
Nếu mạng hoặc cache không sẵn sàng, notebook chuyển sang trọng số khởi tạo và
ghi cảnh báo.

Lưu ý: VGG-16 có khoảng 138M tham số, còn RepLKNet-31B có khoảng 79M. Vì vậy
đây là phép so sánh khác biệt kiến trúc giữa CNN thuần kernel nhỏ và large-kernel
CNN, không phải phép so sánh hai model có cùng capacity.

## Chạy notebook

Mở PowerShell tại thư mục project:

```powershell
.\.venv\Scripts\jupyter lab
```

Mở `notebooks/RepLKNet_Demo_Vietnamese.ipynb` và chạy từ trên xuống dưới. Nên
đặt ảnh thật vào `images/` trước phần 8. Nếu thư mục trống, notebook tự tạo
`images/demo_shapes.png` để kiểm tra pipeline; ảnh hình học này không phải ảnh
ImageNet nên prediction chỉ có tính minh họa.

## Notebook gồm những phần nào?

0. Giới thiệu project.
1. Kiểm tra Python, PyTorch, CUDA, GPU và VRAM.
2. WHAT – RepLKNet là gì?
3. WHY – vì sao kernel lớn và receptive field hữu ích?
4. RepLKNet vs CNN thuần truyền thống với VGG-16.
5. HOW – kiến trúc, hai nhánh và structural re-parameterization.
6. Load RepLKNet-31B pretrained.
7. Load VGG-16 pretrained làm baseline CNN thuần.
8. Inference cùng một ảnh và bảng Top-1/Top-5.
9. Benchmark parameters, kích thước, latency và GPU memory.
10. ERF trực quan bằng gradient, có lưu ảnh vào `results/erf/`.
11. Tổng kết kết quả demo.
12. Scope project đề xuất cho sinh viên.

Các code cell đều có markdown tiếng Việt giải thích mục đích, lý do chạy và kết
quả mong đợi ngay phía trước.

## Thực nghiệm trên Flowers Recognition

Phần thực nghiệm dùng bộ [Flowers Recognition trên Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition), gồm 5 lớp hoa. Kaggle mô tả dataset khoảng 4.242 ảnh; archive đang được tải trong project có 4.317 ảnh được `ImageFolder` đọc hợp lệ, với số lượng từng lớp được ghi cụ thể trong `split_manifest.json`. Mục tiêu là so sánh khả năng transfer learning của **RepLKNet-31B** và **VGG-16 (CNN thuần)** trên cùng dữ liệu, không dùng prediction của ảnh demo.

Trang dữ liệu ghi license là `Unknown`; việc sử dụng dataset cho báo cáo hoặc công bố cần kiểm tra lại điều kiện phân phối và trích dẫn của Kaggle.

Protocol được cố định trong mã nguồn: split stratified 70/15/15, seed ghi trong artifact, input 224×224, ImageNet normalization, augmentation chỉ ở train, CrossEntropy với label smoothing, AdamW, CosineAnnealingLR, chọn checkpoint theo validation macro-F1 và đánh giá một lần trên test split độc lập. RepLKNet chỉ gọi `structural_reparam()` sau khi train để đo inference.

Lệnh chuẩn bị dataset và chạy một seed:

```powershell
python -m pip install -r requirements.txt
python scripts/run_flower_experiment.py --download-kaggle --epochs 10 --batch-size 8 --seeds 42
```

Notebook báo cáo tương ứng là `notebooks/Flower_Experiment_Vietnamese.ipynb`.

Để có kết luận ổn định hơn, chạy nhiều seed với cùng split:

```powershell
python scripts/run_flower_experiment.py --data-dir data/flowers --epochs 10 --batch-size 8 --seeds 42 123 2024
```

Kết quả được lưu trong `results/flowers/`: `summary.csv`, `aggregate_metrics.json`, `conclusion.md`, lịch sử train, confusion matrix, checkpoint tốt nhất và `split_manifest.json`. Cờ `--allow-random-init` chỉ dành cho smoke test; không dùng kết quả đó cho kết luận khoa học.

## GPU và compatibility

Cell đầu notebook luôn chạy:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Các cell inference/benchmark dùng `torch.inference_mode()`. Benchmark có warm-up,
nhiều lần lặp và `torch.cuda.synchronize()` trước khi ghi nhận thời gian. Cell
ERF cần gradient nên không dùng `no_grad`.

Repository chính thức được viết cho hệ sinh thái PyTorch/timm cũ hơn. Wrapper
không sửa source gốc, chỉ import API chính thức và xử lý checkpoint có tiền tố
`module.` hoặc key `model`/`state_dict`. Nếu `timm` quá mới gây lỗi import,
hãy dùng virtual environment riêng và thử bản `timm` trong `requirements.txt`;
không downgrade toàn bộ Python environment đang dùng cho project khác.

## Kết quả mong đợi

Sau khi chạy notebook với checkpoint và GPU, thư mục `results/` có thể chứa:

- `predictions/predictions.json`: Top-1, Top-5 và confidence của hai model.
- `comparison/benchmark.json`: parameters, model size, latency và VRAM.
- `erf/*.npy`, `erf/*.png`: contribution map và hình ERF.

Latency phụ thuộc mạnh vào GPU, driver, PyTorch, batch size và implementation
large-kernel. Vì vậy bảng benchmark dùng để quan sát trên máy đang chạy, không
phải kết luận tuyệt đối rằng model nào luôn nhanh hơn.

## Nguồn tham khảo

- [Paper: Scaling Up Your Kernels to 31x31](https://arxiv.org/abs/2203.06717)
- [Repository PyTorch chính thức](https://github.com/DingXiaoH/RepLKNet-pytorch)
