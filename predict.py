from ultralytics import YOLO

img_path = 'datasets\data.png'
model_path = r'run\seg_train_2024-02-24\train11\weights\best.pt'
model = YOLO(model_path)
result = model.predict(
    source=img_path,
    save = True,
    save_conf = True,
    project = 'runs/segment'
)