from ultralytics import YOLO

img_path = 'datasets\data.png'
model_path = 'models\seg_2024_01_10.pt'
model = YOLO(model_path)
result = model.predict(
    source=img_path,
    save = True,
    save_conf = True
)