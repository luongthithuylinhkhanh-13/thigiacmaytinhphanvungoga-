import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from ultralytics import YOLO
from chatbot import get_chat_response

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Cấu hình
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('app', 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Giới hạn 16 MB
app.config['LATEST_DETECTION'] = None

# Các định dạng ảnh cho phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# --- LOAD MODEL GLOBALLY ---
MODEL_PATH = 'models/best.pt'
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"[+] Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"[!] WARNING: Model file not found at {MODEL_PATH}. Prediction will fail unless model is provided.")
except Exception as e:
    print(f"[!] Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def entity_too_large(e):
    return jsonify({'success': False, 'error': 'File quá lớn. Vui lòng chọn ảnh dưới 16MB.'}), 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Mô hình AI chưa được tải thành công. Vui lòng đảm bảo file models/best.pt đã tồn tại và khởi động lại server.'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Không tìm thấy file tải lên.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Chưa chọn ảnh nào.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Định dạng file không được hỗ trợ.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        
        try:
            conf_threshold = float(request.form.get('conf_threshold', 0.5))
        except ValueError:
            conf_threshold = 0.5
        conf_threshold = max(0.01, min(0.99, conf_threshold))
        
        try:
            iou_threshold = float(request.form.get('iou_threshold', 0.3))
        except ValueError:
            iou_threshold = 0.3
        iou_threshold = max(0.01, min(0.99, iou_threshold))

        # --- AI INFERENCE (SEGMENTATION) ---
        print(f"[*] Executing YOLO predict with conf={conf_threshold}, iou={iou_threshold}")
        results = model.predict(source=upload_path, save=False, conf=conf_threshold, iou=iou_threshold)
        
        if not results or len(results) == 0:
            return jsonify({'success': False, 'error': 'Lỗi trong quá trình phân tích ảnh (không trả về kết quả).'}), 500
            
        result = results[0]
        
        pothole_count = len(result.boxes) if result.boxes is not None else 0
        confidence = 0.0
        area_text = "N/A"
        potholes_data = []
        
        if pothole_count > 0 and result.boxes is not None:
            confs = result.boxes.conf.cpu().numpy()
            if len(confs) > 0:
                confidence = float(np.mean(confs)) * 100
                
            if result.masks is not None:
                mask_data = result.masks.data.cpu().numpy()
                total_pixels = int(np.sum(mask_data > 0.5))
                area_text = str(total_pixels)
                
                # Tính toán diện tích cho từng ổ gà
                for i in range(len(mask_data)):
                    area = int(np.sum(mask_data[i] > 0.5))
                    conf_val = float(confs[i]) * 100 if i < len(confs) else 0.0
                    potholes_data.append({
                        "id": f"Ổ gà {i+1}",
                        "area": area,
                        "confidence": conf_val
                    })

        # Tính toán mức độ nguy hiểm (Risk Level)
        risk_level = "Thấp (An toàn)"
        if pothole_count > 3 or (area_text != "N/A" and int(area_text) > 100000):
            risk_level = "Cao (Nghiêm trọng - Cần khắc phục gấp)"
        elif pothole_count > 0:
            risk_level = "Trung bình (Cần dặm vá chú ý)"

        # Lưu kết quả phân tích mới nhất vào Flask App Config để RAG Chatbot truy xuất
        app.config['LATEST_DETECTION'] = {
            "pothole_count": pothole_count,
            "confidence": confidence,
            "area": area_text,
            "risk_level": risk_level
        }
        
        res_img_bgr = result.plot()
        
        result_filename = f"result_{unique_filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, res_img_bgr)
        
        original_image_url = url_for('static', filename=f"uploads/{unique_filename}")
        result_image_url = url_for('static', filename=f"results/{result_filename}")
        
        return jsonify({
            'success': True,
            'original_image': original_image_url,
            'result_image': result_image_url,
            'pothole_count': pothole_count,
            'confidence': confidence,
            'area': area_text,
            'potholes_data': potholes_data
        })
        
    except Exception as e:
        print(f"[!] Error during AI prediction: {e}")
        return jsonify({'success': False, 'error': f'Lỗi hệ thống khi xử lý ảnh: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message') or data.get('question') or ''
    
    if not user_message.strip():
        return jsonify({'answer': 'Xin lỗi, không nhận được nội dung câu hỏi.', 'reply': 'Xin lỗi, không nhận được nội dung câu hỏi.'}), 400
    
    # Lấy ngữ cảnh từ request hoặc dữ liệu YOLO vừa phân tích trong Flask
    context = data.get('context')
    if not context and app.config['LATEST_DETECTION']:
        context = app.config['LATEST_DETECTION']
        
    reply = get_chat_response(user_message, context=context)
    
    return jsonify({
        'answer': reply,
        'reply': reply
    })

if __name__ == '__main__':
    app.run(debug=True)
