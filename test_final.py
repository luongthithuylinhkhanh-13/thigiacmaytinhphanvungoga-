import os
import glob
from ultralytics import YOLO
import cv2
import numpy as np
import shutil

def run_test():
    model_path = r"runs\segment\runs\segment\pothole_seg_final\weights\best.pt"
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    val_images = glob.glob(r"archive\Pothole_Segmentation_YOLOv8\valid\images\*.jpg")
    sample_images = val_images[:3]
    
    out_dir = "manual_eval_results"
    artifact_dir = r"C:\Users\LINH\.gemini\antigravity-ide\brain\7f987c8f-84ad-4467-994b-6a4bfcf9a4d7"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n=== RUNNING PREDICTIONS ===")
    for img_path in sample_images:
        base_name = os.path.basename(img_path)
        # Bắt buộc conf=0.5, iou=0.3
        results = model.predict(source=img_path, conf=0.5, iou=0.3, save=False)
        
        result = results[0]
        res_img = result.plot()
        
        out_path = os.path.join(out_dir, base_name)
        cv2.imwrite(out_path, res_img)
        
        shutil.copy(out_path, os.path.join(artifact_dir, f"eval_0.5_{base_name}"))
        
        count = len(result.boxes)
        confs = result.boxes.conf.cpu().numpy()
        avg_conf = np.mean(confs) if count > 0 else 0
        
        print(f"Image: {base_name} | Potholes: {count} | Avg Conf: {avg_conf:.4f}")

    gt_batch = r"runs\segment\runs\segment\pothole_seg_final\val_batch0_labels.jpg"
    pred_batch = r"runs\segment\runs\segment\pothole_seg_final\val_batch0_pred.jpg"
    
    if os.path.exists(gt_batch):
        shutil.copy(gt_batch, os.path.join(artifact_dir, "gt_batch.jpg"))
    if os.path.exists(pred_batch):
        shutil.copy(pred_batch, os.path.join(artifact_dir, "pred_batch.jpg"))

if __name__ == "__main__":
    run_test()
