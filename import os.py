import os
import yaml
from ultralytics import YOLO

BASE_DIR = r"C:\dataset" 

def fix_yaml_pose(dataset_location):
    """
    авто правкп data.yaml под диск C:
    """
    yaml_path = os.path.join(dataset_location, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"ощибка: Файл {yaml_path} не найден")
        return None

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    config['path'] = dataset_location.replace("\\", "/")
    config['train'] = "images"
    config['val'] = "images" 
    
    if 'kpt_shape' not in config:
        config['kpt_shape'] = [17, 3]

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return yaml_path

def train_metal_eye_pose():
    print(f"--- 1. работа с папкой: {BASE_DIR} ---")
    
    config_path = fix_yaml_pose(BASE_DIR)
    if not config_path:
        return

    print("--- 2. веса YOLOv8-Pose... ---")
    model = YOLO('yolov8n-pose.pt') 

    print(f"--- 3. обучение ---")
    model.train(
        data=config_path,    
        epochs=150,          
        imgsz=640,           
        device=0,            
        batch=16,            
        workers=4,          
        project='',  #тут типа имя 
        name='experiment_C_drive',
        exist_ok=True,
        plots=True           
    )

if __name__ == '__main__':
    try:
        train_metal_eye_pose()
    except Exception as e:
        print(f"\n--- ощибка: {e} ---")
