import os
import glob
from PIL import Image

DATASET_DIR = r"d:\thigiacmaytinhphanvungoga\archive\Pothole_Segmentation_YOLOv8"

def check_split(split_name):
    img_dir = os.path.join(DATASET_DIR, split_name, "images")
    lbl_dir = os.path.join(DATASET_DIR, split_name, "labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        return 0, 0, 0, 0
    
    img_files = glob.glob(os.path.join(img_dir, "*.*"))
    lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    
    bad_imgs = 0
    bad_lbls = 0
    
    for img_path in img_files:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            bad_imgs += 1
            
    for lbl_path in lbl_files:
        try:
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    # YOLO seg should have class + at least 3 points (6 coords) -> min 7 parts
                    if len(parts) > 0 and len(parts) < 7:
                        bad_lbls += 1
                        break
                    if len(parts) > 0:
                        # check if coords are roughly between 0 and 1
                        coords = [float(x) for x in parts[1:]]
                        # Just a light check to see it's normalized
                        if any(c < 0.0 or c > 1.05 for c in coords):
                            bad_lbls += 1
                            break
        except Exception:
            bad_lbls += 1
            
    return len(img_files), len(lbl_files), bad_imgs, bad_lbls

train_res = check_split("train")
valid_res = check_split("valid")
test_res = check_split("test")

print("=== DATASET CHECK ===")
print(f"Train: {train_res[0]} images, {train_res[1]} labels (Bad img: {train_res[2]}, Bad lbl: {train_res[3]})")
print(f"Valid: {valid_res[0]} images, {valid_res[1]} labels (Bad img: {valid_res[2]}, Bad lbl: {valid_res[3]})")
print(f"Test: {test_res[0]} images, {test_res[1]} labels (Bad img: {test_res[2]}, Bad lbl: {test_res[3]})")

yaml_path = os.path.join(DATASET_DIR, "data.yaml")
if os.path.exists(yaml_path):
    print("\n--- data.yaml ---")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        print(f.read())
else:
    print("\n--- data.yaml DOES NOT exist ---")
