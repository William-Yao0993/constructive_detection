import comet_ml
comet_ml.init(project_name = 'Rebar_detection')
comet_ml.config.save(api_key = 'Qn88wPyWB3PmP0hTUAwhg1h9u')

import ultralytics
from ultralytics import YOLO
from datetime import datetime
ultralytics.checks()

print ("Experiment Start:", datetime.now())

#Load and Train a model
LOCAL_PATH = 'datasets\YOLODataset\dataset.yaml'
SERVER_PATH = '/mnt/data/dayhoff/home/u6771897/constructive_detection/datasets/YOLODataset/data_dayhoff.yaml'

model = YOLO('yolov8n.yaml').load('yolov8n.pt')
result = model.train(data = SERVER_PATH,
                     epochs = 150,
                     lr0 = 0.0001,
                     lrf = 0.01,
                     save = True)
print("Experiment End:", datetime.now())