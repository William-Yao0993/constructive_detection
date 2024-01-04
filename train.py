import comet_ml
import datetime
experiment = comet_ml.Experiment(
        project_name= "Rabar_Segmentation"
)
experiment.set_name(f"trial{datetime.datetime.now().date().strftime('%Y%m%d')}")
# comet_ml.config.save(api_key = 'Qn88wPyWB3PmP0hTUAwhg1h9u')
import ultralytics
from ultralytics import YOLO
ultralytics.checks()
print ("Experiment Start:", datetime.datetime.now().isoformat())

#Train a model
LOCAL_PATH = 'datasets\dataset.yaml'
SERVER_PATH = '/mnt/data/dayhoff/home/u6771897/constructive_detection/datasets/YOLODataset/data_dayhoff.yaml'

model = YOLO('yolov8n.yaml', task='segment')
result = model.train(data = LOCAL_PATH,
                     epochs = 1,
                     lr0 = 0.0001,
                     lrf = 0.01,
                     batch = 1
                     )
print("Experiment End:", datetime.datetime.now())