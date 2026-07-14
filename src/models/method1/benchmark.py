import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger
import csv
import numpy as np

logger = CustomLogger(name="Benchmark", log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/true_generation_model.log').get_logger()

class Benchmark:

    def __init__(self):

        path = '/home/hajali/Desktop/Bargh_Ml_project/models/benchmark/'
        if os.path.exists(path=path):
            self.path = path
        else:
            os.mkdir(path=path)
            self.path = path
        
        self.generators_file = f"{path}/GenerationPredictors.csv"
        self.status_file = f"{path}/StatusPredictors.csv"
        self.classifiers_file = f"{path}/Classifiers.csv"

        if not os.path.exists(self.generators_file):
            with open(self.generators_file, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id', 'mean train', 'last train', 'mean test', 'last test'])
        
        if not os.path.exists(self.status_file):
            with open(self.status_file, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id', 'mean train', 'last train', 'mean test', 'last test'])
        
        if not os.path.exists(self.classifiers_file):
            with open(self.classifiers_file, 'w') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['id', 'mean train', 'last train', 'mean test', 'last test'])
    
    def add_model(self, train_loss: list[int], test_loss:list[int], id:str, N:int, is_generation_predictor=False, is_status_predictor=False):
        
        try:
            mean_train = np.mean(train_loss[-N:])
            mean_test = np.mean(test_loss[-N:])
            last_train = train_loss[-1]
            last_test = test_loss[-1]

            row = [id, np.round(mean_train, 4), np.round(last_train, 4), np.round(mean_test, 4), np.round(last_test, 4)]

            if is_generation_predictor:
                with open(self.generators_file, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
            
            if is_status_predictor:
                with open(self.status_file, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
            
            logger.info(msg=f"Model with id {id} added to benchmark file.")
        except Exception as e:
            logger.error(msg=f"could not add the model with id {id} to the benchmark file. Exception below occured:\n{e}")
    
    def add_classifier(self, id, f1_weighted, f1_macro):
        row = [id, f1_weighted, f1_macro]

        with open(self.classifiers_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        
        logger.info(msg=f"Model with id {id} added to the benchmark file.")
