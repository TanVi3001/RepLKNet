# Kết quả chạy notebook

Notebook có thể tạo các file sau khi chạy:

- `predictions/predictions.json`: kết quả Top-1/Top-5.
- `comparison/benchmark.json`: parameters, kích thước, latency và GPU memory.
- `erf/*.npy`, `erf/*.png`: contribution map và hình ERF.

Các file kết quả phụ thuộc checkpoint, ảnh đầu vào, phiên bản thư viện và phần
cứng. Vì vậy không dùng một lần chạy CPU với trọng số khởi tạo để kết luận model
nào tốt hơn. Hãy chạy lại notebook với checkpoint chính thức nếu cần số liệu có
ý nghĩa cho thuyết trình.
