# RepLKNet: What, Why so vs CNN, How và demo so sánh

Tài liệu này dùng để thuyết trình, viết báo cáo và chạy demo cho bài báo:

> Xiaohan Ding et al., *Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs*, arXiv:2203.06717v4, 2022.

## 1. What - RepLKNet là gì?

RepLKNet là một kiến trúc **CNN thuần** được thiết kế để sử dụng các convolution kernel rất lớn, tối đa **31x31**, thay vì chủ yếu xếp chồng nhiều kernel 3x3 như CNN truyền thống.

Tên RepLKNet có thể hiểu là *Re-parameterized Large-Kernel Network*:

- **Large-kernel:** dùng kernel không gian lớn, điển hình theo bốn stage là `[31, 29, 27, 13]` ở RepLKNet-31.
- **Depth-wise convolution:** kernel lớn chủ yếu xử lý từng channel riêng, nên chi phí tăng ít hơn nhiều so với convolution đầy đủ giữa mọi channel.
- **Re-parameterized:** trong lúc train, một nhánh kernel nhỏ 3x3 chạy song song với kernel lớn để giúp tối ưu; sau khi train, nhánh nhỏ và BatchNorm được gộp vào kernel lớn.
- **Shortcut:** residual/identity shortcut giữ thông tin chi tiết và giúp mạng sâu với kernel lớn ổn định hơn.

Vì vậy, RepLKNet không thay CNN bằng Transformer. Nó trả lời câu hỏi: **CNN có thể mở rộng receptive field và thu thập ngữ cảnh xa bằng kernel lớn hay không?**

### Receptive field và effective receptive field

Receptive field lý thuyết là vùng đầu vào có thể ảnh hưởng tới một neuron ở feature map. Với chuỗi kernel 3x3, receptive field tăng dần theo độ sâu. RepLKNet đưa vùng quan sát rộng vào từng block bằng kernel lớn hơn.

Effective receptive field (ERF) là phần receptive field thực sự có ảnh hưởng mạnh trong quá trình lan truyền gradient. Bài báo cho thấy ERF của RepLKNet phân bố rộng hơn so với ResNet nhỏ-kernel có độ sâu lớn tương tự. Điều này giúp model nhìn được quan hệ giữa nhiều bộ phận của vật thể và bối cảnh xung quanh.

### Thông điệp ngắn để trình bày

> RepLKNet vẫn là CNN, nhưng thay cách tích lũy ngữ cảnh bằng rất nhiều kernel 3x3 bằng một số depth-wise kernel lớn, được re-parameterize để train ổn định và inference gọn.

## 2. Why so vs CNN - Vì sao dùng kernel lớn thay CNN truyền thống?

Ở đây “vs CNN” nên hiểu là **RepLKNet so với CNN truyền thống dùng small kernel**, ví dụ VGG-16 hoặc các backbone chủ yếu dùng 3x3. RepLKNet cũng thuộc họ CNN.

### Vấn đề của CNN nhỏ-kernel

CNN nhỏ-kernel có locality bias mạnh: mỗi layer chỉ nhìn vùng lân cận. Để lấy thông tin xa, mạng phải:

1. xếp chồng nhiều layer;
2. tăng độ sâu hoặc dùng downsampling;
3. truyền thông tin qua nhiều bước trung gian.

Cách này hiệu quả và dễ tối ưu, nhưng ERF thực tế có thể vẫn tập trung quanh vị trí đang xét. Model dễ dựa vào texture cục bộ hơn là hình dạng tổng thể. Với detection, segmentation hoặc ảnh có nhiều ngữ cảnh, đây là một hạn chế.

### Lợi ích của kernel lớn

| Khía cạnh | CNN nhỏ-kernel | RepLKNet |
|---|---|---|
| Vùng nhìn mỗi layer | Hẹp, thường 3x3 | Rộng, tối đa 31x31 |
| Cách lấy ngữ cảnh xa | Qua nhiều layer | Gom trực tiếp trong large kernel |
| ERF | Tập trung hơn | Rộng và phân bố hơn |
| Thiên lệch biểu diễn | Dễ thiên về texture | Tăng shape bias theo quan sát của bài báo |
| Khả năng transfer | Tốt cho classification, có thể thiếu context | Đặc biệt hữu ích cho downstream tasks |
| Inference | Kernel nhỏ dễ được thư viện hỗ trợ | Cần triển khai large DW convolution tối ưu |

Bài báo báo cáo các kết quả quan trọng sau:

- Trên MobileNetV2, thay 3x3 bằng 13x13 có shortcut tăng ImageNet top-1 từ **71.76% lên 72.53%**.
- Không có shortcut, kernel lớn làm kết quả giảm mạnh, cho thấy shortcut là thành phần cần thiết.
- Trong thí nghiệm RepLKNet, đổi kernel theo `[3,3,3,3]` sang `[31,29,27,13]` làm số tham số tăng khoảng **10.4%** và FLOPs tăng khoảng **18.6%**; phần lớn chi phí vẫn đến từ các 1x1 convolution.
- Trên ImageNet, RepLKNet-31B đạt **83.5% top-1**, 79M parameters và 15.3G FLOPs ở 224x224. Throughput báo cáo là 295.5 images/s trên GTX 2080Ti với batch 64 FP32.
- Ở downstream semantic segmentation ADE20K, cấu hình `[31,29,27,13]` đạt 49.17 mIoU, cao hơn cấu hình `[3,3,3,3]` đạt 46.05 mIoU trong bảng ablation của bài báo.

### Vì sao depth-wise convolution làm kernel lớn khả thi?

Với convolution đầy đủ, số tham số gần tỉ lệ với `Cin x Cout x K x K`. Với depth-wise convolution, chi phí không còn nhân chéo giữa toàn bộ channel mà gần tỉ lệ với `C x K x K`. Vì vậy kernel 31x31 vẫn có thể dùng trong backbone nếu phần point-wise 1x1 và triển khai GPU được tối ưu.

Tuy nhiên, FLOPs thấp không tự động có nghĩa là latency thấp. Bài báo chỉ ra implementation PyTorch thông thường hỗ trợ large depth-wise convolution chưa tốt, nên RepLKNet cần kernel CUDA/implementation được tối ưu. Khi đánh giá thực tế phải đo latency trên đúng GPU, batch size, precision và backend.

### Vì sao cần re-parameterization?

Kernel rất lớn có thể khó tối ưu, đặc biệt khi dữ liệu hoặc training budget nhỏ. Trong training, block có dạng khái niệm:

```text
input -> large DW conv -> BN --+
                                +--> activation --> output
input -> small 3x3 conv -> BN -+
```

Sau training, nhánh 3x3 được zero-pad vào giữa kernel lớn và các BatchNorm được fuse. Khi đó:

```text
input -> một large DW conv đã gộp -> activation -> output
```

Model inference giữ được hàm biến đổi tương đương nhưng không còn phải chạy hai nhánh. Đây là lý do chữ “Rep” trong RepLKNet.

## 3. How - RepLKNet hoạt động như thế nào?

### Luồng xử lý

```text
Ảnh đầu vào
    |
Stem / downsampling
    |
Stage 1: RepLK Blocks, kernel lớn
    |
Downsampling
    |
Stage 2: RepLK Blocks, kernel lớn
    |
Downsampling
    |
Stage 3: RepLK Blocks, kernel lớn
    |
Downsampling
    |
Stage 4: RepLK Blocks, kernel lớn hơn kích thước không gian còn lại
    |
Global pooling -> classifier
```

Một RepLK Block kết hợp các thành phần chính:

1. point-wise convolution 1x1 để trộn channel;
2. large depth-wise convolution để trộn thông tin theo không gian;
3. BatchNorm và activation;
4. identity shortcut;
5. point-wise convolution/FFN-style projection để tạo biểu diễn mới.

Mỗi stage được mô tả bởi số block `B`, số channel `C` và kernel size `K`. RepLKNet-31 trong bài báo dùng kernel size theo stage là `[31,29,27,13]` với cấu hình block `[2,2,18,2]` và channel `[128,256,512,1024]` trong thí nghiệm chính.

### Quy trình train và inference

1. Khởi tạo backbone từ pretrained ImageNet nếu có.
2. Thay classifier cuối theo số lớp của bài toán mới.
3. Train với large kernel và nhánh 3x3 re-parameterization.
4. Chọn checkpoint theo validation metric.
5. Fuse các nhánh để structural re-parameterization.
6. Đo test metric và latency trên checkpoint tốt nhất.

Điểm cần nhớ: **fuse sau khi chọn checkpoint**, không fuse sớm rồi dùng nhầm model train-time để so sánh.

## 4. Demo so sánh với CNN truyền thống trong repo

Demo hiện tại dùng:

- **VGG-16** làm đại diện CNN truyền thống small-kernel;
- **RepLKNet-31B** làm large-kernel CNN;
- Flowers Recognition gồm 5 lớp: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`;
- cùng split stratified 70% train, 15% validation, 15% test;
- ảnh 224x224, ImageNet normalization, pretrained backbone;
- cùng optimizer/training protocol;
- ba seed `42, 123, 2024`.

Notebook để trình diễn đầy đủ là [`Flower_Experiment_Vietnamese.ipynb`](../notebooks/Flower_Experiment_Vietnamese.ipynb). Notebook demo một ảnh ngoài dataset là [`Flower_External_Image_Light_Demo_Vietnamese.ipynb`](../notebooks/Flower_External_Image_Light_Demo_Vietnamese.ipynb).

### Chạy thực nghiệm từ terminal

```powershell
python scripts/run_flower_experiment.py `
  --data-dir data/flowers `
  --device cuda `
  --epochs 10 `
  --batch-size 8 `
  --seeds 42 123 2024
```

Nếu chỉ muốn kiểm tra pipeline nhanh, có thể dùng 3 epoch như artifacts hiện tại. Kết quả dùng để kết luận nên dùng checkpoint pretrained và protocol đầy đủ; `--allow-random-init` chỉ phù hợp smoke test.

### Kết quả demo hiện có

Các số dưới đây được lấy từ `results/flowers/aggregate_metrics.json`, với 3 seed và 3 epoch:

| Model | Accuracy | Macro-F1 | Balanced accuracy | Latency / ảnh |
|---|---:|---:|---:|---:|
| VGG-16 | 0.7761 ± 0.0620 | 0.7731 ± 0.0655 | 0.7758 ± 0.0564 | 14.042 ± 1.944 ms |
| RepLKNet-31B | 0.9469 ± 0.0103 | 0.9461 ± 0.0105 | 0.9444 ± 0.0102 | 28.603 ± 3.994 ms |

Chênh lệch trung bình là **+17.08 điểm phần trăm accuracy** và **+17.30 điểm phần trăm macro-F1** cho RepLKNet-31B. Trong run hiện tại, RepLKNet chính xác hơn nhưng latency cao hơn VGG-16; vì thế khi triển khai phải cân bằng accuracy, tốc độ, VRAM và batch size.

### Cách trình diễn demo trong 3 phút

1. Mở notebook thực nghiệm và xác nhận hai model dùng cùng split.
2. Giải thích VGG-16 là baseline CNN 3x3, RepLKNet là CNN 31x31 depth-wise.
3. Chạy bảng metric và biểu đồ `results/flowers/comparison_metrics.png`.
4. Mở một confusion matrix của mỗi model để chỉ ra lớp nào còn nhầm.
5. Chạy notebook ảnh ngoài dataset để cho người xem thấy hai model dự đoán cùng một input.
6. Kết luận bằng macro-F1 trên test set; ảnh demo và latency chỉ là minh họa bổ sung.

## 5. Cách trả lời ngắn khi thuyết trình

### What?

RepLKNet là CNN large-kernel. Nó dùng depth-wise convolution tối đa 31x31, residual shortcut và structural re-parameterization. Mục tiêu là mở rộng ERF và học ngữ cảnh/hình dạng tốt hơn CNN 3x3.

### Why so vs CNN?

CNN truyền thống cũng có thể nhìn xa bằng cách xếp nhiều layer 3x3, nhưng ERF thực tế thường tập trung và model dễ thiên về texture. Kernel lớn đưa context vào sớm hơn. Depth-wise convolution giữ chi phí ở mức chấp nhận được, còn shortcut và re-parameterization giúp train ổn định.

### How?

Train-time dùng large-kernel branch song song với nhánh 3x3 và BN. Sau khi train, fuse nhánh nhỏ vào kernel lớn để inference bằng một branch. Backbone đi qua nhiều stage với kernel `[31,29,27,13]`, sau đó pooling và classifier.

## 6. Giới hạn khi diễn giải kết quả

- RepLKNet và VGG-16 có capacity, số tham số và thiết kế tổng thể khác nhau; không được quy toàn bộ chênh lệch accuracy chỉ cho kernel size.
- Artifacts hiện tại dùng 3 epoch để kiểm tra pipeline và so sánh sample-efficiency. Báo cáo cuối nên dùng training schedule dài hơn và ghi rõ cấu hình.
- Latency phụ thuộc GPU, phiên bản PyTorch/CUDA, batch size, precision và trạng thái fuse.
- Một ảnh demo không thay thế đánh giá trên test set độc lập.
- Kết luận trong repo chỉ có giá trị cho dataset, split, seed, preprocessing và hardware đã ghi trong artifacts.

## Tài liệu tham chiếu

1. [`2203.06717v4.pdf`](../2203.06717v4.pdf) - bài báo gốc.
2. [Official RepLKNet repository](https://github.com/DingXiaoH/RepLKNet-pytorch).
3. [`results/flowers/conclusion.md`](../results/flowers/conclusion.md) - kết luận thực nghiệm.
4. [`results/flowers/aggregate_metrics.json`](../results/flowers/aggregate_metrics.json) - metric gốc theo mean/std/CI.
