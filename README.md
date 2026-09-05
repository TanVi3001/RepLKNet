# RepLKNet (2022) vs CNN truyền thống trên Flowers Recognition

Project tập trung vào một nhiệm vụ duy nhất: huấn luyện và đánh giá RepLKNet-31B so với VGG-16 trên bộ dữ liệu Flowers Recognition gồm 5 lớp hoa. VGG-16 được dùng làm CNN thuần truyền thống; hai mô hình dùng cùng ảnh, cùng cách chia dữ liệu và cùng quy trình đánh giá.

## Cấu trúc project

```text
RepLKNet/
├── data/flowers/                         # Dataset Flowers Recognition
├── notebooks/
│   ├── Flower_Experiment_Vietnamese.ipynb             # Thực nghiệm chính, nhiều seed
│   └── Flower_External_Image_Light_Demo_Vietnamese.ipynb # Nhận diện ảnh ngoài dataset
├── RepLKNet-pytorch/                     # Source RepLKNet chính thức
├── src/
│   ├── flower_experiment.py             # Dataset, train, metrics, checkpoint
│   └── replknet_demo.py                  # Wrapper RepLKNet và tiện ích inference
├── scripts/run_flower_experiment.py     # Chạy thực nghiệm từ terminal
├── weights/RepLKNet-31B_ImageNet-1K_224.pth
├── results/flowers/                      # Kết quả đã chạy và artifact khoa học
├── requirements.txt
└── README.md
```

`RepLKNet-pytorch/` được giữ nguyên vì `src/replknet_demo.py` import trực tiếp implementation chính thức từ file `replknet.py`. Các thư mục demo ImageNet, ERF độc lập và ảnh hình học đã được loại khỏi project.

## Môi trường hiện tại

Project đã được kiểm tra với virtual environment `.venv`, PyTorch CUDA và GPU NVIDIA RTX 3050 6GB. Có thể kiểm tra lại bằng PowerShell:

```powershell
cd E:\NCKH\DL\CNN\RepLKNet\RepLKNet
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu cần tạo lại môi trường:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset và protocol

Dataset được đọc bằng `ImageFolder`, gồm 5 lớp: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`. Protocol hiện tại:

- chia stratified 70% train, 15% validation, 15% test;
- giữ cùng split cho hai mô hình và ghi split vào `results/flowers/split_manifest.json`;
- ảnh đầu vào 224×224, ImageNet normalization;
- augmentation chỉ áp dụng cho train;
- chọn checkpoint theo validation macro-F1, sau đó đánh giá đúng một lần trên test;
- báo cáo accuracy, macro-F1, weighted-F1, precision, recall, confusion matrix, số tham số, thời gian và VRAM;
- chạy nhiều seed để báo cáo mean ± standard deviation thay vì kết luận từ một lần chạy.

Ảnh dùng cho nhận diện ngoài mẫu có thể đặt trong `external_images/` hoặc truyền bằng `EXTERNAL_IMAGE_PATH`/biến môi trường `REPLKNET_EXTERNAL_IMAGE`. Notebook kiểm tra ảnh không thuộc `data/flowers/`, không có trong split manifest và không trùng SHA-256 với bất kỳ ảnh nào trong dataset trước khi chạy cả hai model. Ảnh này chỉ dùng cho inference, không được đưa vào train/test hoặc metrics.

## Chạy thực nghiệm đầy đủ

```powershell
cd E:\NCKH\DL\CNN\RepLKNet\RepLKNet
.\.venv\Scripts\python.exe scripts/run_flower_experiment.py `
  --data-dir data/flowers `
  --device cuda `
  --epochs 10 `
  --batch-size 8 `
  --seeds 42 123 2024
```

Kết quả được lưu trong `results/flowers/`, gồm `summary.csv`, `aggregate_metrics.json`, `conclusion.md`, lịch sử train, confusion matrix, checkpoint tốt nhất và manifest của split.

## Hai notebook chính

1. `Flower_Experiment_Vietnamese.ipynb`: thực nghiệm chính để lấy metrics có ý nghĩa khoa học giữa RepLKNet và VGG-16 trên test set độc lập.
2. `Flower_External_Image_Light_Demo_Vietnamese.ipynb`: nhận diện một ảnh hoa hoàn toàn bên ngoài train/test bằng cả hai model.

Mở notebook bằng:

```powershell
.\.venv\Scripts\jupyter.exe lab
```

Trong VS Code, chọn kernel `replknet-demo` hoặc interpreter `E:\NCKH\DL\CNN\RepLKNet\RepLKNet\.venv\Scripts\python.exe`.

## Checkpoint RepLKNet

Checkpoint ImageNet của RepLKNet-31B được đặt tại `weights/RepLKNet-31B_ImageNet-1K_224.pth`. Nếu thiếu file, có thể tải từ nguồn chính thức trong repository của tác giả hoặc dùng hàm download đã có trong wrapper. Khi fine-tune trên Flowers, phần classifier được thay bằng 5 lớp và checkpoint fine-tuned được lưu trong `results/flowers/`.

## Tài liệu tham khảo

- [Paper: Scaling Up Your Kernels to 31x31](https://arxiv.org/abs/2203.06717)
- [Repository RepLKNet chính thức](https://github.com/DingXiaoH/RepLKNet-pytorch)
- [Flowers Recognition trên Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
