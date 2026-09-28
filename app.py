"""
PotholeSeg - Hệ thống phát hiện và phân vùng ổ gà trên mặt đường
Sử dụng YOLOv8 Segmentation + Flask
"""

import os
import uuid
import time
import json
import sqlite3
from datetime import datetime, timedelta
from functools import wraps

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, session, flash, send_file, g
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.datastructures import FileStorage
from io import BytesIO
import urllib.request
from urllib.parse import urlparse
from PIL import Image

# ============================================================
# Cấu hình Flask App
# ============================================================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'potholeseg_secret_key_2024'

# Đường dẫn lưu trữ
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
DATABASE = 'potholeseg.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB cho video

# Định dạng file cho phép
ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'}
ALLOWED_VIDEO_EXT = {'mp4', 'avi', 'mov'}

# Tạo thư mục cần thiết
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================================
# Load YOLOv8 Model
# ============================================================
MODEL_PATH = 'models/best.pt'
model = None
DEMO_MODE = True

try:
    from ultralytics import YOLO
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        DEMO_MODE = False
        print(f"[+] Model loaded: {MODEL_PATH}")
    else:
        print(f"[!] Model not found: {MODEL_PATH} -> DEMO MODE")
except ImportError:
    print("[!] ultralytics not installed -> DEMO MODE")
except Exception as e:
    print(f"[!] Error loading model: {e} -> DEMO MODE")

# ============================================================
# Database
# ============================================================
def get_db():
    """Lấy kết nối database cho request hiện tại"""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    """Đóng kết nối database khi kết thúc request"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Khởi tạo database và tạo bảng"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Bảng users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Bảng analysis_history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            original_path TEXT,
            result_path TEXT,
            pothole_count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            total_area INTEGER DEFAULT 0,
            processing_time REAL DEFAULT 0,
            status TEXT DEFAULT 'completed',
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("[+] Database initialized")

# Khởi tạo database khi app khởi động
init_db()

# ============================================================
# Decorator: Yêu cầu đăng nhập
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vui lòng đăng nhập để tiếp tục.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Lấy thông tin user hiện tại từ session"""
    if 'user_id' in session:
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        return user
    return None

# ============================================================
# Helper functions
# ============================================================
def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext

def generate_unique_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"

# ============================================================
# Error Handlers
# ============================================================
@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def entity_too_large(e):
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': False, 'error': 'File quá lớn.'}), 413
    flash('File quá lớn. Vui lòng chọn file nhỏ hơn.', 'error')
    return redirect(request.referrer or url_for('index'))

# ============================================================
# Routes: Trang công khai
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ============================================================
# Routes: Xác thực (Authentication)
# ============================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validate
        errors = []
        if not full_name:
            errors.append('Họ và tên không được để trống.')
        if not email:
            errors.append('Email không được để trống.')
        if not password:
            errors.append('Mật khẩu không được để trống.')
        elif len(password) < 6:
            errors.append('Mật khẩu phải có ít nhất 6 ký tự.')
        if password != confirm_password:
            errors.append('Xác nhận mật khẩu không khớp.')

        if errors:
            for err in errors:
                flash(err, 'error')
            return render_template('register.html', full_name=full_name, email=email)

        # Kiểm tra email đã tồn tại
        db = get_db()
        existing = db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            flash('Email đã được sử dụng.', 'error')
            return render_template('register.html', full_name=full_name, email=email)

        # Tạo tài khoản
        password_hash = generate_password_hash(password)
        db.execute(
            'INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)',
            (full_name, email, password_hash)
        )
        db.commit()

        flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember')

        if not email or not password:
            flash('Vui lòng nhập email và mật khẩu.', 'error')
            return render_template('login.html', email=email)

        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['full_name']
            session['user_email'] = user['email']

            if remember:
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=30)

            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Email hoặc mật khẩu không đúng.', 'error')
            return render_template('login.html', email=email)

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Đã đăng xuất thành công.', 'success')
    return redirect(url_for('index'))

# ============================================================
# Routes: Dashboard (Yêu cầu đăng nhập)
# ============================================================
@app.route('/dashboard')
@login_required
def dashboard():
    user = get_current_user()
    db = get_db()

    # Thống kê
    stats = {
        'total_images': db.execute(
            'SELECT COUNT(*) FROM analysis_history WHERE user_id = ? AND file_type = ?',
            (session['user_id'], 'image')
        ).fetchone()[0],
        'total_videos': db.execute(
            'SELECT COUNT(*) FROM analysis_history WHERE user_id = ? AND file_type = ?',
            (session['user_id'], 'video')
        ).fetchone()[0],
        'total_potholes': db.execute(
            'SELECT COALESCE(SUM(pothole_count), 0) FROM analysis_history WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()[0],
        'avg_time': db.execute(
            'SELECT COALESCE(AVG(processing_time), 0) FROM analysis_history WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()[0]
    }

    # Phân tích gần đây
    recent = db.execute(
        'SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 5',
        (session['user_id'],)
    ).fetchall()

    # Dữ liệu biểu đồ: ổ gà theo ngày (7 ngày gần nhất)
    chart_data = db.execute('''
        SELECT DATE(created_at) as date, SUM(pothole_count) as count
        FROM analysis_history
        WHERE user_id = ? AND created_at >= DATE('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    ''', (session['user_id'],)).fetchall()

    chart_labels = [row['date'] for row in chart_data]
    chart_values = [row['count'] for row in chart_data]

    return render_template('dashboard/index.html',
                           user=user, stats=stats, recent=recent,
                           chart_labels=json.dumps(chart_labels),
                           chart_values=json.dumps(chart_values),
                           demo_mode=DEMO_MODE)

# ============================================================
# Routes: Nhận diện ảnh
# ============================================================
@app.route('/detect-image', methods=['GET', 'POST'])
@login_required
def detect_image():
    user = get_current_user()
    if request.method == 'POST':
        file_obj = None
        filename = ''
        image_url = request.form.get('image_url')

        if image_url:
            parsed = urlparse(image_url)
            if parsed.scheme not in ['http', 'https']:
                return jsonify({'success': False, 'error': 'Chỉ hỗ trợ URL http/https.'}), 400
            if parsed.hostname in ['localhost', '127.0.0.1', '::1'] or (parsed.hostname and (parsed.hostname.startswith('192.168.') or parsed.hostname.startswith('10.'))):
                return jsonify({'success': False, 'error': 'URL không hợp lệ.'}), 400

            try:
                req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status != 200:
                        return jsonify({'success': False, 'error': 'Không thể tải ảnh từ URL.'}), 400
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('image/'):
                        return jsonify({'success': False, 'error': 'URL không chứa ảnh hợp lệ.'}), 400
                    
                    file_data = response.read()
                    if len(file_data) > app.config['MAX_CONTENT_LENGTH']:
                        return jsonify({'success': False, 'error': 'Ảnh từ URL quá lớn.'}), 400
                    
                    ext = content_type.split('/')[-1]
                    if ext == 'jpeg': ext = 'jpg'
                    filename = os.path.basename(parsed.path)
                    if not filename or '.' not in filename:
                        filename = f"downloaded.{ext}"
                    
                    file_obj = FileStorage(stream=BytesIO(file_data), filename=filename, content_type=content_type)
            except Exception as e:
                return jsonify({'success': False, 'error': f'Lỗi tải URL: {str(e)}'}), 400
        else:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'Không tìm thấy file.'}), 400
            file_obj = request.files['file']
            filename = file_obj.filename

        if not file_obj or filename == '':
            return jsonify({'success': False, 'error': 'Chưa chọn file.'}), 400

        if not allowed_file(filename, ALLOWED_IMAGE_EXT):
            return jsonify({'success': False, 'error': 'Chỉ hỗ trợ JPG, PNG, WEBP, BMP, TIFF.'}), 400

        try:
            original_name = filename
            unique_name = generate_unique_filename(filename)
            upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file_obj.save(upload_path)
            
            # Validate and convert image using PIL
            try:
                img = Image.open(upload_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Overwrite as standard JPG for YOLO
                unique_name = f"{uuid.uuid4().hex}.jpg"
                upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
                img.save(upload_path, 'JPEG')
            except Exception as e:
                return jsonify({'success': False, 'error': 'File không phải là ảnh hợp lệ hoặc bị hỏng.'}), 400

            start_time = time.time()

            # Lấy tham số từ form
            show_mask = request.form.get('show_mask', 'true') == 'true'
            show_contour = request.form.get('show_contour', 'true') == 'true'
            show_conf = request.form.get('show_conf', 'true') == 'true'
            conf_threshold = float(request.form.get('conf_threshold', 0.5))

            if DEMO_MODE:
                # DEMO MODE: Tạo kết quả minh họa
                result_data = generate_demo_result(upload_path, unique_name)
            else:
                # REAL MODE: Sử dụng YOLOv8
                result_data = run_yolo_inference(upload_path, unique_name,
                                                  conf_threshold, show_mask,
                                                  show_contour, show_conf)

            processing_time = round(time.time() - start_time, 3)
            result_data['processing_time'] = processing_time

            # Lưu vào lịch sử
            db = get_db()
            db.execute('''
                INSERT INTO analysis_history
                (user_id, file_name, file_type, original_path, result_path,
                 pothole_count, avg_confidence, total_area, processing_time, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'], original_name, 'image',
                f'uploads/{unique_name}', result_data.get('result_path', ''),
                result_data.get('pothole_count', 0),
                result_data.get('avg_confidence', 0),
                result_data.get('total_area', 0),
                processing_time,
                json.dumps(result_data.get('potholes', []))
            ))
            db.commit()

            sz_bytes = os.path.getsize(upload_path)
            w = result_data.get('image_size', {}).get('width', 0)
            h = result_data.get('image_size', {}).get('height', 0)
            
            response_data = {
                'success': True,
                'pothole_count': result_data.get('pothole_count', 0),
                'average_confidence': result_data.get('avg_confidence', 0),
                'total_damage_area': result_data.get('union_area', 0),
                'damage_percentage': result_data.get('damage_ratio', 0),
                'processing_time': processing_time,
                'original_url': url_for('static', filename=f'uploads/{unique_name}'),
                'overlay_url': url_for('static', filename=result_data.get('result_path', '')),
                'mask_url': url_for('static', filename=result_data.get('mask_path', '')),
                'image_info': {
                    'filename': original_name,
                    'width': w,
                    'height': h,
                    'resolution': f"{w} × {h}",
                    'total_pixels': w * h,
                    'file_size_bytes': sz_bytes,
                    'file_size_mb': round(sz_bytes / (1024 * 1024), 2)
                },
                'model_info': {
                    'name': 'YOLOv8 Segmentation' if not DEMO_MODE else 'YOLOv8 (DEMO)',
                    'task': 'Instance Segmentation',
                    'confidence_threshold': conf_threshold,
                    'iou_threshold': 0.7
                },
                'speed': result_data.get('speed', {'preprocess': 0, 'inference': 0, 'postprocess': 0}),
                'detections': result_data.get('potholes', []),
                'demo_mode': DEMO_MODE
            }
            
            print("\n===== RESPONSE DATA =====")
            import pprint
            pprint.pprint(response_data)
            print("upload exists:", os.path.exists(upload_path), upload_path)
            result_full_path = os.path.join(app.root_path, 'static', result_data.get('result_path', ''))
            print("result exists:", os.path.exists(result_full_path), result_full_path)
            
            return jsonify(response_data)

        except Exception as e:
            print(f"[!] Error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Lỗi xử lý: {str(e)}'}), 500

    return render_template('dashboard/detect_image.html', user=user, demo_mode=DEMO_MODE)

def run_yolo_inference(image_path, unique_name, conf_threshold, show_mask, show_contour, show_conf):
    """Chạy YOLOv8 segmentation inference trên ảnh"""
    start_time = time.perf_counter()
    results = model.predict(source=image_path, save=False, conf=conf_threshold)

    if not results:
        return {'pothole_count': 0, 'avg_confidence': 0, 'total_area': 0, 'potholes': []}

    result = results[0]
    
    print("\n===== YOLO DEBUG =====")
    print("orig_shape:", result.orig_shape)
    print("boxes:", getattr(result, 'boxes', None))
    print("masks:", getattr(result, 'masks', None))
    print("speed:", getattr(result, 'speed', None))

    if getattr(result, 'boxes', None) is not None:
        print("boxes count:", len(result.boxes))
        print("conf:", result.boxes.conf)
        print("cls:", result.boxes.cls)
        print("xyxy:", result.boxes.xyxy)

    if getattr(result, 'masks', None) is not None:
        print("mask count:", len(result.masks.data))
        print("mask shape:", result.masks.data.shape)
        
    # 1. Khởi tạo
    orig_h, orig_w = result.orig_shape
    detections = []
    total_damage_area = 0
    damage_percentage = 0
    
    # BƯỚC 3 & 4: Tính Box và Confidence
    confidences = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0].cpu().item())
            class_id = int(box.cls[0].cpu().item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            confidences.append(conf)
            
            detections.append({
                "id": i + 1,
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": conf,
                "pixel_area": 0,
                "image_percentage": 0,
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1)
                },
                "center": {
                    "x": round(center_x, 1),
                    "y": round(center_y, 1)
                }
            })

    average_confidence = (sum(confidences) / len(confidences)) if confidences else 0

    # BƯỚC 5 & 6: Mask / Pixel Area
    combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    if result.masks is not None:
        mask_data = result.masks.data
        for i, mask in enumerate(mask_data):
            mask_np = mask.cpu().numpy()
            resized_mask = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0.5)
            pixel_area = int(binary_mask.sum())
            
            if i < len(detections):
                detections[i]["pixel_area"] = pixel_area
                detections[i]["image_percentage"] = (pixel_area / (orig_w * orig_h)) * 100
                
            combined_mask = np.logical_or(combined_mask, binary_mask)
            
        total_damage_area = int(np.count_nonzero(combined_mask))
        damage_percentage = (total_damage_area / (orig_w * orig_h)) * 100

    pothole_count = len(detections)

    # 7. Tốc độ
    speed_raw = getattr(result, 'speed', {}) or {}
    preprocess_ms = float(speed_raw.get("preprocess", 0))
    inference_ms = float(speed_raw.get("inference", 0))
    postprocess_ms = float(speed_raw.get("postprocess", 0))
    
    speed_info = {
        "preprocess": preprocess_ms,
        "inference": inference_ms,
        "postprocess": postprocess_ms,
        "total": preprocess_ms + inference_ms + postprocess_ms
    }

    # BƯỚC 14: Draw image
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (orig_w, orig_h))

    # 1. Image mask only
    mask_only_img = np.zeros_like(img) if img is not None else np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    if img is not None:
        mask_only_img[combined_mask > 0] = [255, 150, 0]
    mask_filename = f"mask_{unique_name}"
    mask_path = os.path.join(RESULT_FOLDER, mask_filename)
    cv2.imwrite(mask_path, mask_only_img)

    # 2. Image overlay
    if result.masks is not None and show_mask and img is not None:
        colored_mask = np.zeros_like(img)
        colored_mask[:, :, 0] = 255  # Blue
        colored_mask[:, :, 1] = 100  # Green
        img = np.where(combined_mask[:, :, np.newaxis] > 0,
                       cv2.addWeighted(img, 0.65, colored_mask, 0.35, 0),
                       img)

    if result.masks is not None and show_contour and img is not None:
        for mask in result.masks.data.cpu().numpy():
            mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            binary = (mask_resized > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 255), 2)

    if result.boxes is not None and show_conf and img is not None:
        for d in detections:
            x1, y1, x2, y2 = map(int, [d['bbox']['x1'], d['bbox']['y1'], d['bbox']['x2'], d['bbox']['y2']])
            conf_val = round(d['confidence'] * 100, 1)
            label1 = f"Pothole #{d['id']}"
            label2 = f"Conf: {conf_val}%"
            (w1, h1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (w2, h2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            box_w = max(w1, w2) + 10
            box_h = h1 + h2 + 15
            
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, max(0, y1 - box_h - 10)), (x1 + box_w, max(0, y1 - 10)), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            
            cv2.putText(img, label1, (x1 + 5, max(15, y1 - h2 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img, label2, (x1 + 5, max(15, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    result_filename = f"result_{unique_name}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    if img is not None:
        cv2.imwrite(result_path, img)
    
    return {
        'pothole_count': pothole_count,
        'avg_confidence': average_confidence,
        'union_area': total_damage_area,
        'damage_ratio': damage_percentage,
        'potholes': detections,
        'result_path': f'results/{result_filename}',
        'mask_path': f'results/{mask_filename}',
        'speed': speed_info,
        'image_size': {'width': orig_w, 'height': orig_h}
    }

def generate_demo_result(image_path, unique_name):
    """Tạo kết quả minh họa khi không có model"""
    import random
    img = cv2.imread(image_path)
    if img is None:
        return {'pothole_count': 0, 'avg_confidence': 0, 'total_area': 0, 'union_area': 0, 'damage_ratio': 0, 'potholes': [], 'speed': {'preprocess': 0, 'inference': 0, 'postprocess': 0}, 'image_size': {'width': 0, 'height': 0}}

    h, w = img.shape[:2]
    pothole_count = random.randint(1, 4)
    potholes = []
    total_area = 0

    overlay = img.copy()
    for i in range(pothole_count):
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 3, 2 * h // 3)
        rx = random.randint(30, min(100, w // 4))
        ry = random.randint(20, min(70, h // 4))
        conf = round(random.uniform(65, 95), 1)
        area = int(3.14 * rx * ry)
        total_area += area

        # Vẽ mask ellipse bán trong suốt
        cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, (255, 100, 50), -1)
        # Vẽ đường viền
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 255), 2)
        # Vẽ label
        label = f"Pothole {conf}%"
        cv2.putText(img, label, (cx - rx, cy - ry - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        potholes.append({
            'id': i + 1,
            'class_id': 0,
            'class_name': "Pothole",
            'pixel_area': area,
            'image_percentage': round((area / (w * h)) * 100, 2),
            'confidence': conf / 100,
            'bbox': {'x1': cx-rx, 'y1': cy-ry, 'x2': cx+rx, 'y2': cy+ry},
            'center': {'x': cx, 'y': cy}
        })

    # Blend overlay
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    # Thêm watermark DEMO
    cv2.putText(img, "DEMO MODE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    result_filename = f"result_{unique_name}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, img)

    avg_conf = round(sum(p['confidence'] for p in potholes) / len(potholes), 3) if potholes else 0

    return {
        'pothole_count': pothole_count,
        'avg_confidence': avg_conf,
        'total_area': total_area,
        'union_area': total_area,
        'damage_ratio': round((total_area / (w * h)) * 100, 2),
        'potholes': potholes,
        'result_path': f'results/{result_filename}',
        'mask_path': f'results/{result_filename}',
        'speed': {'preprocess': 1.5, 'inference': 45.2, 'postprocess': 2.1},
        'image_size': {'width': w, 'height': h}
    }

# ============================================================
# Routes: Nhận diện video
# ============================================================
@app.route('/detect-video', methods=['GET', 'POST'])
@login_required
def detect_video():
    user = get_current_user()
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Không tìm thấy file.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Chưa chọn file.'}), 400

        if not allowed_file(file.filename, ALLOWED_VIDEO_EXT):
            return jsonify({'success': False, 'error': 'Chỉ hỗ trợ MP4, AVI, MOV.'}), 400

        try:
            original_name = file.filename
            unique_name = generate_unique_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(upload_path)

            start_time = time.time()

            if DEMO_MODE:
                result_data = generate_demo_video_result(upload_path, unique_name)
            else:
                result_data = run_yolo_video_inference(upload_path, unique_name)

            processing_time = round(time.time() - start_time, 3)
            result_data['processing_time'] = processing_time

            # Lưu lịch sử
            db = get_db()
            db.execute('''
                INSERT INTO analysis_history
                (user_id, file_name, file_type, original_path, result_path,
                 pothole_count, avg_confidence, processing_time, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'], original_name, 'video',
                f'uploads/{unique_name}', result_data.get('result_path', ''),
                result_data.get('pothole_count', 0),
                result_data.get('avg_confidence', 0),
                processing_time,
                json.dumps({'fps': result_data.get('fps', 0),
                            'total_frames': result_data.get('total_frames', 0)})
            ))
            db.commit()

            return jsonify({
                'success': True,
                'result_video': url_for('static', filename=result_data.get('result_path', '')),
                'pothole_count': result_data.get('pothole_count', 0),
                'avg_confidence': result_data.get('avg_confidence', 0),
                'fps': result_data.get('fps', 0),
                'total_frames': result_data.get('total_frames', 0),
                'processing_time': processing_time,
                'demo_mode': DEMO_MODE
            })

        except Exception as e:
            print(f"[!] Video Error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Lỗi xử lý video: {str(e)}'}), 500

    return render_template('dashboard/detect_video.html', user=user, demo_mode=DEMO_MODE)

def run_yolo_video_inference(video_path, unique_name):
    """Chạy YOLOv8 trên video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    result_name = f"result_{unique_name.rsplit('.', 1)[0]}.mp4"
    result_path = os.path.join(RESULT_FOLDER, result_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (w, h))

    total_potholes = 0
    all_confs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, conf=0.5, verbose=False)
        if results:
            result = results[0]
            count = len(result.boxes) if result.boxes is not None else 0
            total_potholes += count
            if result.boxes is not None and count > 0:
                confs = result.boxes.conf.cpu().numpy()
                all_confs.extend(confs.tolist())
            annotated = result.plot()
            out.write(annotated)
        else:
            out.write(frame)

    cap.release()
    out.release()

    avg_conf = round(float(np.mean(all_confs)) * 100, 1) if all_confs else 0

    return {
        'pothole_count': total_potholes,
        'avg_confidence': avg_conf,
        'fps': round(fps, 1),
        'total_frames': total_frames,
        'result_path': f'results/{result_name}'
    }

def generate_demo_video_result(video_path, unique_name):
    """Kết quả minh họa cho video"""
    import random
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    cap.release()

    return {
        'pothole_count': random.randint(5, 20),
        'avg_confidence': round(random.uniform(70, 90), 1),
        'fps': round(fps, 1),
        'total_frames': total_frames,
        'result_path': f'uploads/{unique_name}'
    }

# ============================================================
# Routes: Lịch sử phân tích
# ============================================================
@app.route('/history')
@login_required
def history():
    user = get_current_user()
    db = get_db()

    # Lọc và phân trang
    file_type = request.args.get('type', 'all')
    search = request.args.get('search', '')
    page = int(request.args.get('page', 1))
    per_page = 10

    query = 'SELECT * FROM analysis_history WHERE user_id = ?'
    params = [session['user_id']]

    if file_type != 'all':
        query += ' AND file_type = ?'
        params.append(file_type)

    if search:
        query += ' AND file_name LIKE ?'
        params.append(f'%{search}%')

    # Đếm tổng
    count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
    total = db.execute(count_query, params).fetchone()[0]
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Lấy dữ liệu phân trang
    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
    params.extend([per_page, (page - 1) * per_page])
    records = db.execute(query, params).fetchall()

    return render_template('dashboard/history.html',
                           user=user, records=records,
                           page=page, total_pages=total_pages,
                           file_type=file_type, search=search)

@app.route('/history/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_history(record_id):
    db = get_db()
    record = db.execute(
        'SELECT * FROM analysis_history WHERE id = ? AND user_id = ?',
        (record_id, session['user_id'])
    ).fetchone()

    if record:
        # Xóa file kết quả nếu tồn tại
        if record['result_path']:
            result_file = os.path.join('static', record['result_path'])
            if os.path.exists(result_file):
                os.remove(result_file)
        if record['original_path']:
            original_file = os.path.join('static', record['original_path'])
            if os.path.exists(original_file):
                os.remove(original_file)

        db.execute('DELETE FROM analysis_history WHERE id = ?', (record_id,))
        db.commit()

    return jsonify({'success': True})

@app.route('/history/download/<int:record_id>')
@login_required
def download_result(record_id):
    db = get_db()
    record = db.execute(
        'SELECT * FROM analysis_history WHERE id = ? AND user_id = ?',
        (record_id, session['user_id'])
    ).fetchone()

    if record and record['result_path']:
        file_path = os.path.join('static', record['result_path'])
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)

    flash('Không tìm thấy file kết quả.', 'error')
    return redirect(url_for('history'))

# ============================================================
# Routes: Báo cáo thống kê
# ============================================================
@app.route('/statistics')
@login_required
def statistics():
    user = get_current_user()
    db = get_db()
    uid = session['user_id']

    stats = {
        'total_images': db.execute(
            'SELECT COUNT(*) FROM analysis_history WHERE user_id = ? AND file_type = ?', (uid, 'image')
        ).fetchone()[0],
        'total_videos': db.execute(
            'SELECT COUNT(*) FROM analysis_history WHERE user_id = ? AND file_type = ?', (uid, 'video')
        ).fetchone()[0],
        'total_potholes': db.execute(
            'SELECT COALESCE(SUM(pothole_count), 0) FROM analysis_history WHERE user_id = ?', (uid,)
        ).fetchone()[0],
        'total_analyses': db.execute(
            'SELECT COUNT(*) FROM analysis_history WHERE user_id = ?', (uid,)
        ).fetchone()[0],
        'avg_time': db.execute(
            'SELECT COALESCE(AVG(processing_time), 0) FROM analysis_history WHERE user_id = ?', (uid,)
        ).fetchone()[0]
    }

    # Biểu đồ: ổ gà theo ngày (30 ngày)
    pothole_by_day = db.execute('''
        SELECT DATE(created_at) as date, SUM(pothole_count) as count
        FROM analysis_history WHERE user_id = ? AND created_at >= DATE('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY date
    ''', (uid,)).fetchall()

    # Biểu đồ: ảnh/video theo ngày
    analysis_by_day = db.execute('''
        SELECT DATE(created_at) as date,
               SUM(CASE WHEN file_type='image' THEN 1 ELSE 0 END) as images,
               SUM(CASE WHEN file_type='video' THEN 1 ELSE 0 END) as videos
        FROM analysis_history WHERE user_id = ? AND created_at >= DATE('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY date
    ''', (uid,)).fetchall()

    # Phân bố kích thước ổ gà
    size_dist = db.execute('''
        SELECT total_area FROM analysis_history
        WHERE user_id = ? AND total_area > 0
    ''', (uid,)).fetchall()

    # Thời gian xử lý trung bình theo ngày
    time_by_day = db.execute('''
        SELECT DATE(created_at) as date, AVG(processing_time) as avg_time
        FROM analysis_history WHERE user_id = ? AND created_at >= DATE('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY date
    ''', (uid,)).fetchall()

    return render_template('dashboard/statistics.html',
                           user=user, stats=stats,
                           pothole_by_day=json.dumps([dict(r) for r in pothole_by_day]),
                           analysis_by_day=json.dumps([dict(r) for r in analysis_by_day]),
                           size_dist=json.dumps([dict(r) for r in size_dist]),
                           time_by_day=json.dumps([dict(r) for r in time_by_day]))

# ============================================================
# Routes: Thông tin mô hình
# ============================================================
@app.route('/model-info')
@login_required
def model_info():
    user = get_current_user()

    info = {
        'model_name': 'YOLOv8 Segmentation',
        'model_file': 'best.pt',
        'model_exists': os.path.exists(MODEL_PATH),
        'demo_mode': DEMO_MODE,
        'input_size': '640x640',
        'epochs': 'Chưa có dữ liệu',
        'batch_size': 'Chưa có dữ liệu',
        'precision': 'Chưa có dữ liệu',
        'recall': 'Chưa có dữ liệu',
        'map50': 'Chưa có dữ liệu',
        'map50_95': 'Chưa có dữ liệu',
        'iou': 'Chưa có dữ liệu',
    }

    # Nếu model đã load, thử lấy thông tin
    if model and not DEMO_MODE:
        try:
            info['input_size'] = '640x640'
            # Thông tin từ model metadata (nếu có)
            if hasattr(model, 'ckpt') and model.ckpt:
                ckpt = model.ckpt
                if 'train_args' in ckpt:
                    args = ckpt['train_args']
                    info['epochs'] = str(args.get('epochs', 'Chưa có dữ liệu'))
                    info['batch_size'] = str(args.get('batch', 'Chưa có dữ liệu'))
                    info['input_size'] = f"{args.get('imgsz', 640)}x{args.get('imgsz', 640)}"
        except Exception:
            pass

    return render_template('dashboard/model_info.html', user=user, info=info)

# ============================================================
# Routes: Hướng dẫn sử dụng (trong dashboard)
# ============================================================
@app.route('/dashboard/guide')
@login_required
def dashboard_guide():
    user = get_current_user()
    return render_template('dashboard/guide.html', user=user)

# ============================================================
# Routes: Quản lý tài khoản
# ============================================================
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = get_current_user()

    if request.method == 'POST':
        action = request.form.get('action')
        db = get_db()

        if action == 'update_info':
            full_name = request.form.get('full_name', '').strip()
            if full_name:
                db.execute('UPDATE users SET full_name = ? WHERE id = ?',
                           (full_name, session['user_id']))
                db.commit()
                session['user_name'] = full_name
                flash('Cập nhật thông tin thành công!', 'success')
            else:
                flash('Họ tên không được để trống.', 'error')

        elif action == 'change_password':
            current_pw = request.form.get('current_password', '')
            new_pw = request.form.get('new_password', '')
            confirm_pw = request.form.get('confirm_password', '')

            if not check_password_hash(user['password_hash'], current_pw):
                flash('Mật khẩu hiện tại không đúng.', 'error')
            elif len(new_pw) < 6:
                flash('Mật khẩu mới phải có ít nhất 6 ký tự.', 'error')
            elif new_pw != confirm_pw:
                flash('Xác nhận mật khẩu không khớp.', 'error')
            else:
                new_hash = generate_password_hash(new_pw)
                db.execute('UPDATE users SET password_hash = ? WHERE id = ?',
                           (new_hash, session['user_id']))
                db.commit()
                flash('Đổi mật khẩu thành công!', 'success')

        return redirect(url_for('profile'))

    return render_template('dashboard/profile.html', user=user)

# ============================================================
# API: Download ảnh kết quả (từ detect-image)
# ============================================================
@app.route('/api/download-result')
@login_required
def api_download_result():
    path = request.args.get('path', '')
    if path:
        # Bảo mật: chỉ cho phép download từ results/
        safe_path = os.path.join('static', 'results', os.path.basename(path))
        if os.path.exists(safe_path):
            return send_file(safe_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  PotholeSeg - Pothole Detection System")
    print(f"  Mode: {'DEMO' if DEMO_MODE else 'PRODUCTION'}")
    print(f"  Model: {MODEL_PATH}")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
