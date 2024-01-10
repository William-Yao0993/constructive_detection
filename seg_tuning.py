from comet_ml import Experiment
import datetime
experiment = Experiment(
    project_name='rebar_seg_tuning'
)
experiment.set_name(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

from ultralytics import YOLO
LOCAL_PATH = 'datasets\constructive_seg\dataset.yaml'
SERVER_PATH = '/mnt/data/dayhoff/home/u6771897/constructive_detection/datasets/constructive_seg/data_dayhoff.yaml'
model_path = 'models/seg_2024_01_10.pt'
model = YOLO(model_path,task = 'segment')
result = model.tune(
    data = SERVER_PATH,
    epochs = 100,
    iterations = 300,
    lr0 = 1E-3,
    lrf = 1E-6,
    plots = True,
    save = True,
    val = False,
    augment = True
)