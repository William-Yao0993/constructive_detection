from ultralytics import YOLO

img_path = 'datasets\data.png'
model_path = r'seg_train_2024-01-10\train3\weights\best.pt'
model = YOLO(model_path)
result = model.predict(
    source=img_path,
    save = True,
    save_conf = True,
    classes = 1
)