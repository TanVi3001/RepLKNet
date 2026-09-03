# Kết luận thực nghiệm Flowers Recognition

Protocol đã chạy với 3 seed: 42, 123, 2024.
Mỗi run dùng 3 epoch, batch size 8 và learning rate 0.0003.
Hai model dùng cùng split stratified 70/15/15, cùng preprocessing 224×224, loss CrossEntropy có label smoothing, AdamW và lịch CosineAnnealingLR.

## Kết quả trung bình theo seed

| Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) | Balanced accuracy (mean ± std) |
|---|---:|---:|---:|
| VGG-16 (CNN thuần) | 0.7761 ± 0.0620 | 0.7731 ± 0.0655 | 0.7758 ± 0.0564 |
| RepLKNet-31B | 0.9469 ± 0.0103 | 0.9461 ± 0.0105 | 0.9444 ± 0.0102 |

95% CI xấp xỉ của macro-F1 VGG-16 (CNN thuần): ± 0.0741.
95% CI xấp xỉ của macro-F1 RepLKNet-31B: ± 0.0119.

Run artifact hiện tại dùng 3 epoch để kiểm chứng pipeline và so sánh sample-efficiency; kết luận cuối cho báo cáo nên dùng run dài hơn sau khi kiểm tra đường cong hội tụ.

## Chênh lệch ghép cặp theo seed

RepLKNet-31B − VGG-16: accuracy trung bình +0.1708, macro-F1 trung bình +0.1730 (độ lệch chuẩn macro-F1 0.0589).

Theo macro-F1 trung bình trên test split hiện tại, model đạt kết quả cao hơn là **RepLKNet-31B**.
Kết luận này chỉ có giá trị trong phạm vi dataset, split, protocol và phần cứng đã ghi trong artifacts.
Không dùng latency hoặc một ảnh demo để thay thế đánh giá trên test set.
VGG-16 có nhiều tham số hơn RepLKNet-31B; chênh lệch accuracy không được quy trực tiếp cho large kernel nếu chưa kiểm soát capacity bằng một thí nghiệm bổ sung.
