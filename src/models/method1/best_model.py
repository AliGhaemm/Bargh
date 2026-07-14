import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger
import json
import pandas as pd
import shutil
from true_generation_model import Model

logger = CustomLogger(name="BestModel", log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/best_model.log').get_logger()

class BestModel:

    def __init__(self):
        try:
            self.regressor_path = "/home/hajali/Desktop/Bargh_Ml_project/models/best_models.json"
            self.classifier_path = "/home/hajali/Desktop/Bargh_Ml_project/models/best_classifiers.json"
            self.generationBM = "/home/hajali/Desktop/Bargh_Ml_project/models/benchmark/GenerationPredictors.csv"
            self.classifierBM = "/home/hajali/Desktop/Bargh_Ml_project/models/benchmark/Classifiers.csv"

            if not os.path.exists(self.regressor_path):
                file = open(self.regressor_path, "w")
                file.close()
                logger.info(msg=f"Best model json file did not exist. So created.")
            
            if not os.path.exists(self.classifier_path):
                file = open(self.classifier_path, "w")
                file.close()
                logger.info(msg=f"Best model json file did not exist. So created.")
            
            self.generationBenchmark = pd.read_csv(self.generationBM)
            self.classifierBenchmark = pd.read_csv(self.classifierBM)

            logger.info(msg=f"Class initiated successfully")

        except Exception as e:
            logger.error(msg=f"Could not instanciate the class. Exception below occured:\n{e}")
    
    def update(self):
        try:
            bm = self.generationBenchmark
            powerplants = list(bm['id'].apply(lambda x: '-'.join(x.split('-')[:-1])))

            data = {}

            for powerplant in powerplants:
                sample = bm[bm['id'].str.contains(powerplant, na=False)]
                sample.sort_values(by='last test', ascending=True, inplace=True)
                data[powerplant] = sample.iloc[0]['id']
            
            with open(self.regressor_path, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(msg=f"Update process applied.")
        except Exception as e:
            logger.error(msg=f"Could not apply the update. Exception below occured:\n{e}")
    
    def update_classifiers(self):
        try:
            bm = self.classifierBenchmark
            powerplants = list(bm['id'].apply(lambda x: '-'.join(x.split('-')[:-4])))
            methods = ['generation', 'features']
            clusters = ['gaussian', 'kmeans']

            data = {}
            
            # for powerplant in powerplants:
            #     for method in methods:
            #         for cluster in clusters:
            #             sample = bm[(bm['id'].str.contains(powerplant, na=False)) & (bm['id'].str.contains(method, na=False)) & (bm['id'].str.contains(cluster, na=False))]
            #             if len(sample.index) == 0:
            #                 continue
            #             sample.sort_values(by='f1 weighted', ascending=False, inplace=True)
            #             data[f"{powerplant}-{method}-{cluster}"] = sample.iloc[0]['id']

            for powerplant in powerplants:
                sample = bm[(bm['id'].str.contains(powerplant, na=False))]
                if len(sample.index) == 0:
                    continue
                sample.sort_values(by='f1 weighted', ascending=False, inplace=True)
                data[f"{powerplant}"] = sample.iloc[0]['id']
            
            with open(self.classifier_path, 'w') as f:
                json.dump(data, f, indent=8)

        except Exception as e:
            logger.error(msg=f"Could not apply the update. Exception below occured:\n{e}")
    
    def choose_best_classifier(self, name, code):
        try:
            base_path = "/home/hajali/Desktop/Bargh_Ml_project/models/classifiers/"
            bm = self.update_benchmark()
            units = bm[(bm['id'].str.contains(name, na=False)) & (bm['id'].str.contains(code, na=False))]
            units.sort_values(by='f1 weighted', ascending=False, inplace=True)
            keeping_model = units.iloc[0]['id']
            target_str = f"{name}-{code}"

            for root, dirs, _ in os.walk(base_path, topdown=False):
                for dirname in dirs:
                    if target_str in dirname:
                        if dirname != keeping_model:
                            dir_path = os.path.join(root, dirname)
                            logger.warning(msg=f"Deleting directory: {dir_path}")
                            shutil.rmtree(dir_path)
                        else:
                            logger.info(msg=f"Keeping the model: {dirname}")
        except Exception as e:
            logger.error(msg=f"Exception: {e}")
    
    def best_prediction(self, input:pd.DataFrame):
        try:
            # id = "-".join([str(input.iloc[0]['name']), str(input.iloc[0]['code'])]) if byName is False else "-".join([str(input.iloc[0]['name']), "None"])
            id1 = "-".join([str(input.iloc[0]['name']), str(input.iloc[0]['code'])])
            id2 = "-".join([str(input.iloc[0]['name']), "None"])
            with open(self.regressor_path, 'r') as f:
                data = json.load(f)
            
            if (id1 not in data.keys()) & (id2 not in data.keys()):
                raise Exception("no model for this powerplant have been trined.")
            
            df = pd.read_csv(self.generationBM)
            print("#############################")
            print("CSV file imported into datafram.")
            print("#############################")
            print(df.head())

            model_name1 = data[id1]
            model_name2 = data[id2]
            error1 = df.loc[df['id']==model_name1, 'last test'].values[0]
            error2 = df.loc[df['id']==model_name2, 'last test'].values[0]

            if error1 <= error2:
                model_name = model_name1
            else:
                model_path = f"/home/hajali/Desktop/Bargh_Ml_project/models/{model_name2}"
                y_pred = Model.predict(input=input, model_path=model_path)
                error3 = Model.evaluate_unit(input=input, predictions=y_pred)
                model_name = model_name2 if error3 <= error1 else model_name1
            
            model_path = f"/home/hajali/Desktop/Bargh_Ml_project/models/{model_name}" if error1 <= error2 else f"/home/hajali/Desktop/Bargh_Ml_project/models/{model_name}"
            y_pred = Model.predict(input=input, model_path=model_path) if "Polynomial" not in model_name else Model.predict(input=input, model_path=model_path, is_poly=True)
            print("##########################")
            print(f"model is: {model_name}")
            print(f"#########################")
            return y_pred
        except Exception as e:
            logger.error(msg=f"Could not predict. Exception below occured:\n{e}")
    
    def update_benchmark(self):
        self.classifierBenchmark = pd.read_csv(self.classifierBM)
        return self.classifierBenchmark