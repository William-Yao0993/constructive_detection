import comet_ml
import datetime
# comet_ml.config.save(api_key = 'Qn88wPyWB3PmP0hTUAwhg1h9u')
# experiment = comet_ml.Experiment(
#         project_name= "Rebar_Segmentation"
# )
# experiment.set_name(f"trial{datetime.datetime.now().date().strftime('%Y%m%d')}")
# import os 
# os.environ["OMP_NUM_THREADS"] = '8'


from ultralytics import YOLO
#ultralytics.checks()

def main():
        #Train a model
        LOCAL_PATH = 'datasets\inner_panel\YOLODataset\dataset.yaml'
        SERVER_PATH = r'datasets/panel/YOLODataset/data_dayhoff.yaml'

        model = YOLO(model='yolov8n.pt')
        result = model.train(data=LOCAL_PATH,
                        epochs = 300,
                        patience = 50,
                        lr0 = 1E-3,
                        lrf = 1E-6,
                        optimizer = 'AdamW',
                        save = True,
                        plots = True,
                        val = True,
                        augment = True,
                        project = f'run/det_train_{datetime.date.today().isoformat()}',
                        device = 0
                        )
        
if __name__ == '__main__':
        #torch.multiprocessing.freeze_support()
        main()
