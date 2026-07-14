import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger

logger = CustomLogger(name="Classifier", log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/Classifier.log').get_logger()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from clustering import Cluster
from benchmark import Benchmark
from seasonality import Seasonality

benchmark = Benchmark()

class Classifer:

    def __init__(self, df: pd.DataFrame, target: str, base_path:str, features_to_drop=None):
        if features_to_drop is not None:
            self.df = df.drop(columns=features_to_drop)
        else:
            self.df = df
        self.target = target
        self.droped = features_to_drop
        self.base_path = base_path
        self.method = ""
        self.cluster = ""
        self.n = 0
    

    def update_class(self, method, cluster, n):
        self.method = method
        self.cluster = cluster
        self.n = n
    

    def extract_data(self, name, code):
        try:

            cluster = Cluster()
            
            if code is not None:
                df_copy = self.df[(self.df['name'] == name) & (self.df['code'] == code)]
            else:
                df_copy = self.df[self.df['name'] == name]
            print(f"number of data is: {len(df_copy)}")
            logger.info(msg=f"Successfully splitted the {name}-{code} from the original dataset to train on.")

            
            null_columns = df_copy.columns[df_copy.isnull().all()].tolist()
            df_copy = df_copy.drop(columns=null_columns)
            seasonality_cols_base = ['year_effect', 'week_effect', 'hour_effect']
            seasonality_cols = [x for x in seasonality_cols_base if not x in null_columns]

            df_copy = df_copy.dropna()

            df_copy = pd.get_dummies(df_copy, columns=seasonality_cols)

            if len(df_copy.index) < 2:
                logger.warning(msg=f"Length of the dataset is less than 2 and training stopped on it. returning None.")
                return None, None
            
            
            
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy = df_copy.sort_values(by='date')
            
            df_copy.drop(columns=['id', 'hour', 'date', 'name', 'code', 'status'], axis=1, inplace=True)

            match self.cluster:
                case "gaussian":
                    if self.method == "generation":
                        df_labeled = cluster.gaussian_cluster(data=df_copy, n=self.n, on_generation=True, labeling=True, save=True)
                    elif self.method == "features":
                        df_labeled = cluster.gaussian_cluster(data=df_copy, n=self.n, on_generation=False, labeling=True, save=True)
                case "kmeans":
                    if self.method == "generation":
                        df_labeled = cluster.kmeans_cluster(data=df_copy, n=self.n, on_generation=True, labeling=True, save=True)
                    elif self.method == "features":
                        df_labeled = cluster.kmeans_cluster(data=df_copy, n=self.n, on_generation=False, labeling=True, save=True)


            df_labeled = df_labeled.drop(columns=['generation'])

            X = df_labeled.drop(columns=[self.target])
            y = df_labeled[self.target]

            return X, y
        
        except Exception as e:

            logger.error(msg=f"Exception: {e}")
            return None, None
    
    def scale(self, x, do_flat=False):
        scaler = StandardScaler()
        scaled_x = scaler.fit_transform(x)

        if not do_flat:
            logger.info(f"X scaled successfully.")
            return scaled_x, scaler
        else:
            logger.info(f"y scaled successfully")
            logger.info(f"y flattened.")
            return scaled_x.flatten(), scaler
    

    def split_data(self, X, y, test_size=0.2, random_state=42):
        if len(X) < 5:
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            # n = len(X.index)
            # split_point = int(n * 2/3)
            # X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
            # X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]
        
        return (X_train, y_train, X_test, y_test)
    

    def save_model(self, model, y_test, y_pred, path):
        try:
            counter = 1

            while os.path.exists(path=path):
                path = f"{path.split('#')[0]}#{counter}"
                counter += 1
            
            os.mkdir(path=path)

            joblib.dump(model, f"{path}/mdoel.pkl")
            logger.info(msg=f"Model saved in {path}/model.pkl")
            
            evaluation = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
            eval_df = pd.DataFrame(evaluation)
            eval_df.to_csv(f"{path}/evaluation.csv")
            logger.info(msg=f"eval data saved in {path}/evaluation.csv")

            conf_matrix = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(self.n), yticklabels=self.n)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(f"{path}/conf_matrix.png")
            logger.info(msg=f"Confusion matrix saved sucessfully.")

            macro_f1 = eval_df['macro avg']['f1-score']
            weighted_f1 = eval_df['weighted avg']['f1-score']

            id = path.split('/')[-1]

            benchmark.add_classifier(id=id, f1_macro=macro_f1, f1_weighted=weighted_f1)
        
        except Exception as e:
            logger.error(msg=f"Saving canceled. Exception occured:\n{e}")
    
    
    def report(self, y_pred, y_test, save=False):
        if not save:
            # accuracy = accuracy_score(y_test, y_pred)
            # precision = precision_score(y_test, y_pred)
            # recall = recall_score(y_test, y_pred)
            # f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # print(f"Accuracy: {accuracy:.2f}")
            # print(f"Precision: {precision:.2f}")
            # print(f"Recall: {recall:.2f}")
            # print(f"F1 Score: {f1:.2f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
        else:
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict)
            path = "/home/hajali/Desktop/file.csv"
            report_df.to_csv(path)

    
    def logistic(self, name, code):

        try:

            id = f"{name}-{code}-{self.method}-{self.cluster}-{self.n}"
            path = f"{self.base_path}/{id}-logistic#0"

            X, y = self.extract_data(name, code)

            X_train, y_train, X_test, y_test = self.split_data(X, y)

            model = LogisticRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            self.save_model(model=model, y_test=y_test, y_pred=y_pred, path=path)

            del model, X_train, X_test, y_train, y_test, y_pred

        except Exception as e:
            logger.error(msg=f"Training fialed. Exception below occured:\n{e}")
    

    def RFC(self, name, code, n_estimator=100):

        try:

            id = f"{name}-{code}-{self.method}-{self.cluster}-{self.n}"
            path = f"{self.base_path}/{id}-RFC#0"

            X, y = self.extract_data(name, code)

            X_train, y_train, X_test, y_test = self.split_data(X, y)

            model = RandomForestClassifier(n_estimators=n_estimator, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            self.save_model(model=model, y_test=y_test, y_pred=y_pred, path=path)

            del model, X_train, X_test, y_train, y_test, y_pred

        except Exception as e:
            logger.error(msg=f"Training fialed. Exception below occured:\n{e}")


    def LDA(self, name, code):

        try:

            id = f"{name}-{code}-{self.method}-{self.cluster}-{self.n}"
            path = f"{self.base_path}/{id}-LDA#0"

            X, y = self.extract_data(name, code)

            X_train, y_train, X_test, y_test = self.split_data(X, y)

            model = LinearDiscriminantAnalysis()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            self.save_model(model=model, y_test=y_test, y_pred=y_pred, path=path)

            del model, X_train, X_test, y_train, y_test, y_pred
        
        except Exception as e:
            logger.error(msg=f"Training fialed. Exception below occured:\n{e}")


    def QDA(self, name, code):

        try:

            id = f"{name}-{code}-{self.method}-{self.cluster}-{self.n}"
            path = f"{self.base_path}/{id}-QDA#0"

            X, y = self.extract_data(name, code)

            X_train, y_train, X_test, y_test = self.split_data(X, y)

            model = QuadraticDiscriminantAnalysis()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            self.save_model(model=model, y_test=y_test, y_pred=y_pred, path=path)

            del model, X_train, X_test, y_train, y_test, y_pred
        
        except Exception as e:
            logger.error(msg=f"Training fialed. Exception below occured:\n{e}")


    def SMC(self, name, code):

        try:

            id = f"{name}-{code}-{self.method}-{self.cluster}-{self.n}"
            path = f"{self.base_path}/{id}-SMC#0"

            X, y = self.extract_data(name, code)

            X_train, y_train, X_test, y_test = self.split_data(X, y)

            model = SVC(probability=True, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            self.save_model(model=model, y_test=y_test, y_pred=y_pred, path=path)

            del model, X_train, X_test, y_train, y_test, y_pred
        
        except Exception as e:
            logger.error(msg=f"Training fialed. Exception below occured:\n{e}")


    def KNN(self, name, code, n=5):

        try:

            id = f"{name}-{code}-{self.method}-{self.cluster}-{self.n}"
            path = f"{self.base_path}/{id}-KNN#0"

            X, y = self.extract_data(name, code)

            X_train, y_train, X_test, y_test = self.split_data(X, y)

            model = KNeighborsClassifier(n_neighbors=n)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            self.save_model(model=model, y_test=y_test, y_pred=y_pred, path=path)

            del model, X_train, X_test, y_train, y_test, y_pred
        
        except Exception as e:
            logger.error(msg=f"Training fialed. Exception below occured:\n{e}")


    def XGB(self, name, code):

        try:

            id = f"{name}-{code}-{self.method}-{self.cluster}-{self.n}"
            path = f"{self.base_path}/{id}-XGB#0"

            X, y = self.extract_data(name, code)

            X_train, y_train, X_test, y_test = self.split_data(X, y)

            # dtrain = xgb.DMatrix(X_train, label=y_train)
            # dtest = xgb.DMatrix(X_test, label=y_test)

            # params = {
            #     'objective': 'multi:softmax',  
            #     'eval_metric': 'logloss',         
            #     'max_depth': 6,                   
            #     'learning_rate': 0.1,            
            #     'n_estimators': 100,             
            #     'colsample_bytree': 0.8,         
            #     'subsample': 0.8,                
            #     'seed': 42                       
            # }

            # model = xgb.train(params, dtrain, num_boost_round=100)

            model = xgb.XGBClassifier(object="multi:softmax", num_classes=3, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # y_pred_prob = model.predict(dtest)
            # y_pred = (y_pred_prob > 0.5).astype(int)

            self.save_model(model=model, y_test=y_test, y_pred=y_pred, path=path)

            del model, X_train, X_test, y_train, y_test, y_pred
        
        except Exception as e:
            logger.error(msg=f"Training fialed. Exception below occured:\n{e}")


    def MLP(self, name, code, hidden_size=(100,)):

        try:

            id = f"{name}-{code}-{self.method}-{self.cluster}-{self.n}"
            path = f"{self.base_path}/{id}-MLP#0"

            X, y = self.extract_data(name, code)

            X_train, y_train, X_test, y_test = self.split_data(X, y)

            model = MLPClassifier(hidden_layer_sizes=hidden_size, max_iter=200, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            self.save_model(model=model, y_test=y_test, y_pred=y_pred, path=path)

            del model, X_train, X_test, y_train, y_test, y_pred
        
        except Exception as e:
            logger.error(msg=f"Training fialed. Exception below occured:\n{e}")
    
    @ classmethod
    def predict(self, input: pd.DataFrame, model_path, droped_features, return_df=False):
        model = self.load_model(model_path)
        null_columns = input.columns[input.isnull().all()].tolist()
        input = input.drop(columns=null_columns)
        seasonality_cols_base = ['year_effect', 'week_effect', 'hour_effect']
        seasonality_cols = [x for x in seasonality_cols_base if not x in null_columns]

        input = input.dropna()

        input = pd.get_dummies(input, columns=seasonality_cols)
        input_cp = input.drop(columns=['id', 'hour', 'date', 'name', 'code', 'status', 'generation'])
        input_cp = input_cp.drop(columns=droped_features)
        labels = model.predict(input_cp)
        if return_df:
            input['label'] = labels
            return input
        else:
            print(labels)

    @classmethod
    def load_model(self, model_path):
        with open(f"{model_path}/mdoel.pkl", 'rb') as f:
            model = joblib.load(f)
        
        return model