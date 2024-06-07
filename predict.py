from ultralytics import YOLO

img_path = 'datasets\data.png'
model_path = r'models\inner_panel.pt'
model = YOLO(model_path)
result = model.predict(
    source=img_path,
    save = True,
    save_conf = True,
    project = r'run/predict'
)