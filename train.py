import comet_ml
comet_ml.init(project_name = 'Rebar_detection')
# comet_ml.config.save(api_key = 'Qn88wPyWB3PmP0hTUAwhg1h9u')

import ultralytics
from ultralytics import YOLO
from datetime import datetime
ultralytics.checks()
#

from roboflow import Roboflow
rf = Roboflow(api_key="yI73TRx0NvIR8Jg0ZP1h")
project = rf.workspace("anu-mhqfx").project("rebar-6duke")
dataset = project.version(1).download("yolov8")


print ("Experiment Start:", datetime.now())

# #Load and Train a model
# LOCAL_PATH = 'datasets\dataset.yaml'
# SERVER_PATH = '/mnt/data/dayhoff/home/u6771897/constructive_detection/datasets/YOLODataset/data_dayhoff.yaml'

# model = YOLO('yolov8n.yaml').load('yolov8n.pt')
# result = model.train(data = LOCAL_PATH,
#                      epochs = 1,
#                      lr0 = 0.0001,
#                      lrf = 0.01,
#                      batch = 1,
#                      save = True)
# print("Experiment End:", datetime.now())