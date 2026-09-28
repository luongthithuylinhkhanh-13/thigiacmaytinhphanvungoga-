# PotholeSeg - Hệ thống phát hiện và phân vùng ổ gà trên mặt đường

## Giới thiệu

PotholeSeg là hệ thống web ứng dụng thị giác máy tính và mô hình **YOLOv8 Segmentation** để phát hiện và phân vùng ổ gà trên mặt đường.

**Đề tài:** Nghiên cứu xây dựng mô hình thị giác máy tính phân vùng ổ gà trên mặt đường.

## Công nghệ sử dụng

- **Backend:** Python Flask
- **AI Model:** YOLOv8 Segmentation (Ultralytics)
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript
- **Charts:** Chart.js

## Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
python app.py
```

### 3. Truy cập

Mở trình duyệt tại: **http://127.0.0.1:5000**

## Cấu trúc project

```
potholeseg/
├── app.py                  # File chính Flask application
├── requirements.txt        # Dependencies
├── README.md               # Hướng dẫn
├── potholeseg.db           # Database SQLite (tự tạo)
├── models/
│   └── best.pt             # Model YOLOv8 Segmentation
├── static/
│   ├── css/
│   │   └── style.css       # Stylesheet chính
│   ├── uploads/            # Ảnh/video upload
│   └── results/            # Kết quả phân tích
└── templates/
    ├── base.html            # Template gốc
    ├── index.html           # Trang chủ
    ├── login.html           # Đăng nhập
    ├── register.html        # Đăng ký
    ├── about.html           # Giới thiệu
    ├── guide.html           # Hướng dẫn
    ├── contact.html         # Liên hệ
    ├── partials/
    │   └── navbar.html      # Thanh menu
    └── dashboard/
        ├── base.html        # Layout dashboard
        ├── index.html       # Dashboard chính
        ├── detect_image.html # Nhận diện ảnh
        ├── detect_video.html # Nhận diện video
        ├── history.html     # Lịch sử phân tích
        ├── statistics.html  # Báo cáo thống kê
        ├── model_info.html  # Thông tin mô hình
        ├── guide.html       # Hướng dẫn sử dụng
        └── profile.html     # Quản lý tài khoản
```

## Tính năng

### Trang công khai
- Trang chủ với hero section và form đăng ký
- Giới thiệu, Hướng dẫn, Liên hệ
- Đăng ký / Đăng nhập

### Dashboard (sau đăng nhập)
- **Trang chủ:** Thống kê tổng quan, biểu đồ, phân tích gần đây
- **Nhận diện ảnh:** Upload ảnh, phân tích YOLOv8, hiển thị kết quả segmentation
- **Nhận diện video:** Upload video, phân tích từng frame
- **Lịch sử phân tích:** Bảng lịch sử với tìm kiếm, lọc, phân trang
- **Báo cáo thống kê:** Biểu đồ Chart.js chi tiết
- **Thông tin mô hình:** Thông số YOLOv8 Segmentation
- **Quản lý tài khoản:** Cập nhật thông tin, đổi mật khẩu

## DEMO MODE

Nếu file `models/best.pt` không tồn tại hoặc không thể load, hệ thống tự động chuyển sang **DEMO MODE**:
- Kết quả phân tích là dữ liệu minh họa
- Ứng dụng vẫn hoạt động bình thường
- Hiển thị badge "DEMO MODE" trên giao diện

## Lưu ý

- Mật khẩu được hash bằng `werkzeug.security` (không lưu plain-text)
- Database SQLite tự động tạo khi khởi động
- Thông số đánh giá mô hình (Precision, Recall, mAP, IoU) hiển thị "Chưa có dữ liệu" nếu chưa có kết quả evaluate thực tế
