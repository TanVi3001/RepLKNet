# Flower experiment artifacts

Thư mục này dành cho kết quả thực nghiệm trên Flowers Recognition. Lệnh chạy sẽ tạo:

- `experiment_config.json`: dataset, split, seed, hyperparameters, device và phiên bản thư viện.
- `split_manifest.json`: danh sách ảnh và index của train/validation/test.
- `summary.csv` và `aggregate_metrics.json`: bảng metrics theo model và theo seed.
- `comparison_metrics.png`: biểu đồ mean ± standard deviation của các metrics chính.
- `conclusion.md`: kết luận định lượng sinh từ các kết quả đã lưu.
- `seed_<n>/`: history, checkpoint tốt nhất, confusion matrix và metrics chi tiết.

Checkpoint `.pt` được giữ ngoài version control vì kích thước lớn.
