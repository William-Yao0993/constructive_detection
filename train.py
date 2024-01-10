import comet_ml
import datetime
# comet_ml.config.save(api_key = 'Qn88wPyWB3PmP0hTUAwhg1h9u')
experiment = comet_ml.Experiment(
        project_name= "Rebar_Segmentation"
)
experiment.set_name(f"trial{datetime.datetime.now().date().strftime('%Y%m%d')}")
import os 
os.environ["OMP_NUM_THREADS"] = '8'

import ultralytics
from ultralytics import YOLO
ultralytics.checks()
print ("Experiment Start:", datetime.datetime.now().isoformat())

#Train a model
LOCAL_PATH = 'datasets\constructive_seg\dataset.yaml'
SERVER_PATH = '/mnt/data/dayhoff/home/u6771897/constructive_detection/datasets/constructive_seg/data_dayhoff.yaml'

model = YOLO('yolov8l-seg.pt',task = 'segment')
result = model.train(data=SERVER_PATH,
                     epochs = 999,
                     lr0 = 1E-3,
                     lrf = 1E-6,
                     plots = False,
                     val = False,
                     augment = False,
                     project = f'seg_train_{datetime.date.today().isoformat()}'
                     )
print("Experiment End:", datetime.datetime.now())