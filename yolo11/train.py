model_yaml_path = r"datasets\yolo11.yaml"
data_yaml_path = r"data.yaml"

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model_yaml_path)

    model.train(
        data=data_yaml_path,  
        cache=False,
        imgsz=640,
        epochs=100,
        single_cls=False,
        batch=8,
        close_mosaic=10,
        workers=0,
        device='0',  # 用 GPU
        optimizer='SGD',
        amp=True,
        project='runs/train',
        name='exp_yolo11',
    )
