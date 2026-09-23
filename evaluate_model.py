import os
import glob
from ultralytics import YOLO
import cv2
import numpy as np

def evaluate():
    print("=== MODEL EVALUATION ===")
    model_path = 'models/best.pt'
    if not os.path.exists(model_path):
        print("Error: models/best.pt not found")
        return
        
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # 1. Chạy đánh giá (Validation) để lấy chính xác các chỉ số
    print("\n--- RUNNING VALIDATION ---")
    metrics = model.val(data='dataset/data.yaml', split='val', plots=True)
    
    print("\n=== METRICS REPORT ===")
    print(f"Box Precision: {metrics.box.p.mean():.4f}")
    print(f"Box Recall: {metrics.box.r.mean():.4f}")
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    
    print(f"Mask Precision: {metrics.seg.p.mean():.4f}")
    print(f"Mask Recall: {metrics.seg.r.mean():.4f}")
    print(f"Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    
    # 2. Tạo ảnh dự đoán mẫu để người dùng xem trực quan
    print("\n--- GENERATING SAMPLE PREDICTIONS ---")
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    val_images = glob.glob('archive/Pothole_Segmentation_YOLOv8/valid/images/*.jpg')
    # Chọn 5 ảnh đầu tiên làm mẫu
    sample_images = val_images[:5] 
    
    all_confs = []
    
    for i, img_path in enumerate(sample_images):
        # conf=0.25 để cho phép bắt nhiều box để đánh giá False Positive
        results = model.predict(source=img_path, save=False, conf=0.25)
        
        # Lưu ảnh kết quả có vẽ box và mask
        res_img = results[0].plot()
        base_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, f'eval_{base_name}')
        cv2.imwrite(out_path, res_img)
        print(f"Saved prediction for visual inspection: {out_path}")
        
        # Thu thập confidence
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            all_confs.extend(confs)
            
    if all_confs:
        avg_conf = np.mean(all_confs)
        print(f"\nAverage Confidence on Sample Set: {avg_conf:.4f}")
    else:
        print("\nAverage Confidence: No detections on sample set.")
        
    print(f"\nEvaluation complete. Please check the '{output_dir}' folder for visual inspection.")

if __name__ == "__main__":
    evaluate()
