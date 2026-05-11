import os
import yaml
from ultralytics import YOLO

# === НАСТРОЙКИ ===
# Указали твой новый путь, где теперь лежат images и labels
BASE_DIR = r"C:\dataset" 

def fix_yaml_pose(dataset_location):
    """
    Автоматически правит data.yaml под твой диск C:
    """
    yaml_path = os.path.join(dataset_location, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"ОШИБКА: Файл {yaml_path} не найден! Проверь, лежит ли он в корне папки.")
        return None

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Прописываем абсолютные пути, чтобы YOLO не терялась
    config['path'] = dataset_location.replace("\\", "/")
    config['train'] = "images"
    config['val'] = "images" 
    
    # Контрольный: проверяем формат для Pose
    if 'kpt_shape' not in config:
        config['kpt_shape'] = [17, 3]

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return yaml_path

def train_metal_eye_pose():
    print(f"--- 1. Работаем с папкой: {BASE_DIR} ---")
    
    # Исправляем конфиг
    config_path = fix_yaml_pose(BASE_DIR)
    if not config_path:
        return

    # Загружаем модель (специально для скелетов)
    print("--- 2. Загрузка весов YOLOv8-Pose... ---")
    model = YOLO('yolov8n-pose.pt') 

    print(f"--- 3. ПОГНАЛИ! Обучаем MetalEye на твоей 5060 ---")
    model.train(
        data=config_path,    
        epochs=150,          
        imgsz=640,           
        device=0,            # Твоя видюха
        batch=16,            # Если будет ошибка памяти, снизь до 8
        workers=4,           # Оптимально для Windows
        project='MetalEye_Pose',  
        name='experiment_C_drive',
        exist_ok=True,
        plots=True           
    )

if __name__ == '__main__':
    try:
        train_metal_eye_pose()
    except Exception as e:
        print(f"\n--- ПРОИЗОШЛА ОШИБКА: {e} ---")