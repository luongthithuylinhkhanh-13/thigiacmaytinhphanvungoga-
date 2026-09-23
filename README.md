# Hệ Thống Phân Vùng Ổ Gà Trên Mặt Đường (Pothole Segmentation)

Dự án sử dụng Deep Learning (YOLOv8 Segmentation) để phân vùng và nhận diện ổ gà trên mặt đường. Hệ thống cung cấp một ứng dụng web để tải ảnh lên và nhận kết quả trực quan.

## Cấu Trúc Thư Mục

- `app.py`: File chạy chính của server Flask backend.
- `requirements.txt`: Danh sách các thư viện Python cần cài đặt.
- `models/`: Thư mục lưu trữ các file mô hình YOLO (như `best.pt`).
- `dataset/`: Thư mục lưu trữ hoặc liên kết đến dữ liệu gốc.
- `app/templates/`: Chứa mã nguồn giao diện HTML.
- `app/static/`: Chứa CSS, JavaScript và nơi lưu trữ ảnh upload/kết quả.

## Cài Đặt

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi chạy ứng dụng:
   ```bash
   python app.py
   ```
3. Truy cập website tại: `http://127.0.0.1:5000`
