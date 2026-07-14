import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger

logger = CustomLogger(name="classis", log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/true_generation_model.log').get_logger()

from benchmark import Benchmark

bnchmrk = Benchmark()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
import graphviz
from tqdm import tqdm
import joblib
import xgboost as xgb
import os
import time
import json
import pickle
import plotly.graph_objects as go



class TQDMCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_epochs):
        self.pbar = tqdm(total=total_epochs, desc="Training Progress", unit="epoch")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

class MAECallback(xgb.callback.TrainingCallback):
    def __init__(self, dtrain, dtest, y_train_actual, y_test_actual, scaler_y):
        self.dtrain = dtrain
        self.dtest = dtest
        self.y_train_actual = y_train_actual
        self.y_test_actual = y_test_actual
        self.scaler_y = scaler_y
        self.train_mae_per_epoch = []
        self.test_mae_per_epoch = []

    def after_iteration(self, model, epoch, evals_log):
        # Predict on scaled data
        y_pred_train_scaled = model.predict(self.dtrain)
        y_pred_test_scaled = model.predict(self.dtest)

        # Inverse transform to original scale
        y_pred_train_actual = self.scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
        y_pred_test_actual = self.scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

        # Calculate MAE on original scale
        train_mae = mean_absolute_error(self.y_train_actual, y_pred_train_actual)
        test_mae = mean_absolute_error(self.y_test_actual, y_pred_test_actual)

        # Store results
        self.train_mae_per_epoch.append(train_mae)
        self.test_mae_per_epoch.append(test_mae)

        # Return False to indicate training should continue
        return False


class Model:
    def __init__(self, df: pd.DataFrame, target: str, features_to_drop=None):
        if features_to_drop is not None:
            self.df = df.drop(columns=features_to_drop)
        else:
            self.df = df
        self.target = target
        self.droped = features_to_drop
    
    def linear(self, name, code, gd=True, mini_batch=False, penalty=None, epochs=500, initial_lr=0.01, lr="adaptive",
               tol=1e-3, alpha=0.01, batch_size=32, max_iter=1, save_model=False):
        try:

            path = f'/home/hajali/Desktop/Bargh_Ml_project/models/{name}-{code}-linear#0'

            X, y = self.extract_data(name, code)

            X_scaled, scaler_X = self.scale(X)
            y_scaled, scaler_y = self.scale(y.values.reshape(-1, 1), do_flat=True)

            X_train, y_train, X_test, y_test = self.split_data(X_scaled, y_scaled)

            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
            y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
            
            if not gd:
                model = LinearRegression()
                model.fit(X_train, y_train)
                model_info = dict(model.get_params().items())
                y_pred_test= model.predict(X_test)
                y_pred_train = model.predict(X_train)

                y_pred_test_actual = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
                y_pred_train_actual = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

                mse_test_actual = (mean_absolute_error(y_test_actual, y_pred_test_actual) / np.mean(y_test_actual)) * 100
                mse_train_actual = (mean_absolute_error(y_train_actual, y_pred_train_actual) / np.mean(y_train_actual)) * 100

                logger.info(msg=f"Model trained successfully.")
                
                if save_model:
                    path = self.save_model(path=path, model=model, model_info=model_info, scaler_X=scaler_X, scaler_y=scaler_y, train_error=[mse_train_actual], test_error=[mse_test_actual])
                    for f in range(X_train.shape[1]-2):
                        self.plot_model_curve(model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, feature_name=X.columns[f], resolution=50, save_path=path)
                else:
                    print(f"Train Error: {mse_train_actual}%")
                    print(f"Test Error: {mse_test_actual}%")
                    for f in range(X_train.shape[1]-2):
                        self.plot_model_curve(model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, resolution=50)


            else:
                model, model_info = self.create_sgd(
                    penalty=penalty, max_iter=max_iter, tol=tol, alpha=alpha,
                    lr=lr, initial_lr=initial_lr, warm_start=True
                )

                if not mini_batch:

                    test_errors_actual = []
                    train_errors_actual = []

                    model_info["n_epochs"] = epochs

                    for epoch in tqdm(range(epochs), desc='Training Process', position=0, leave=True):
                        model.partial_fit(X_train, y_train)
                        y_pred_test = model.predict(X_test)
                        y_pred_train = model.predict(X_train)

                        y_pred_test_actual = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
                        y_pred_train_actual = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

                        mse_test_actual = mean_absolute_error(y_test_actual, y_pred_test_actual) / np.mean(y_test_actual) * 100
                        mse_train_actual = mean_absolute_error(y_train_actual, y_pred_train_actual) / np.mean(y_train_actual) * 100
                        test_errors_actual.append(mse_test_actual)
                        train_errors_actual.append(mse_train_actual)
                        
                        if (epoch+1)%50 == 0:
                            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Error: {mse_train_actual:.4f}%, Test Error: {mse_test_actual:.4f}%")

                else:
                    test_errors_actual = []
                    train_errors_actual = []

                    model_info["n_epochs"] = epochs
                    model_info["Batch size"] = batch_size

                    for epoch in tqdm(range(epochs), desc="Training Precess", position=0, leave=True):
                        X_train, y_train = shuffle(X_train, y_train, random_state=42)
                        batch_progress = tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}", position=1, leave=False)

                        for i in batch_progress:
                            X_batch = X_train[i:i+batch_size]
                            y_batch = y_train[i:i+batch_size]
                            model.partial_fit(X_batch, y_batch)
                        
                        y_pred_test = model.predict(X_test)
                        y_pred_train = model.predict(X_train)

                        y_pred_test_actual = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
                        y_pred_train_actual = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

                        mse_test_actual = mean_absolute_error(y_test_actual, y_pred_test_actual) / np.mean(y_test_actual) * 100
                        mse_train_actual = mean_absolute_error(y_train_actual, y_pred_train_actual) / np.mean(y_train_actual) * 100
                        test_errors_actual.append(mse_test_actual)
                        train_errors_actual.append(mse_train_actual)
                        

                        if epoch%50 == 0:
                            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Error: {mse_train_actual:.4f}%, Test Error: {mse_test_actual:.4f}%")
                
                logger.info(msg=f"Model trained successfully.")

                if save_model:
                    path = self.save_model(path=path, model=model, model_info=model_info, scaler_X=scaler_X, scaler_y=scaler_y, train_error=train_errors_actual, test_error=test_errors_actual)
                    self.plot_training_process(path=path, epochs=epochs, test_errors=test_errors_actual, train_errors=train_errors_actual, length=len(X_train), save=True)
                    for f in range(X_train.shape[1]-2):
                        self.plot_model_curve(model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, feature_name=X.columns[f], resolution=50, save_path=path)
                else:
                    self.plot_training_process(path=path, epochs=epochs, test_errors=test_errors_actual, train_errors=train_errors_actual, length=len(X_train), save=False)
                    # for f in range(X_train.shape[1]-2):
                    #     self.plot_model_curve(model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, resolution=50)

        except Exception as e:
            logger.error(msg=f"Couldnt train Linear model on {name}-{code} of this dataset. Exception below occured.\n{e}\n")
    
    
    def polynomial(self, name, code, gd=True, degree=2, mini_batch=False, penalty=None, epochs=500, initial_lr=0.01, lr="adaptive",
               tol=1e-3, alpha=0.01, batch_size=32, max_iter=1, save_model=False):
        
        path = f'/home/hajali/Desktop/Bargh_Ml_project/models/{name}-{code}-Polynomial#0'

        X, y = self.extract_data(name, code)

        poly = PolynomialFeatures(degree=degree, include_bias=False)

        X_poly = poly.fit_transform(X)

        X_scaled, scaler_X = self.scale(X_poly)
        y_scaled, scaler_y = self.scale(y.values.reshape(-1, 1), do_flat=True)

        X_train, y_train, X_test, y_test = self.split_data(X_scaled, y_scaled)

        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()

        if not gd:
            model = LinearRegression()
            model.fit(X_train, y_train)
            model_info = dict(model.get_params().items())
            model_info['degree'] = degree
            y_pred_test= model.predict(X_test)
            y_pred_train = model.predict(X_train)

            y_pred_test_actual = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
            y_pred_train_actual = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

            mse_test_actual = mean_absolute_error(y_test_actual, y_pred_test_actual) / np.mean(y_test_actual) * 100
            mse_train_actual = mean_absolute_error(y_train_actual, y_pred_train_actual) / np.mean(y_train_actual) * 100

            logger.info(msg=f"Model trained successfully.")
            
            if save_model:
                path = self.save_model(path=path, model=model, model_info=model_info, scaler_X=scaler_X, scaler_y=scaler_y, train_error=[mse_train_actual], test_error=[mse_test_actual], poly=poly)
                for f in range(X.shape[1]-2):
                    self.plot_model_curve(model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, feature_name=X.columns[f], resolution=50, save_path=path)
            else:
                print(f"Train Error: {mse_train_actual}")
                print(f"Test Error: {mse_test_actual}")

        else:
            model, model_info = self.create_sgd(
                    penalty=penalty, max_iter=max_iter, tol=tol, alpha=alpha,
                    lr=lr, initial_lr=initial_lr, warm_start=True
                )
            model_info['degree'] = degree
            
            if not mini_batch:

                test_errors_actual = []
                train_errors_actual = []

                model_info["n_epochs"]=epochs

                for epoch in tqdm(range(epochs), desc='Training Process', position=0, leave=True):
                    model.partial_fit(X_train, y_train)
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)

                    y_pred_test_actual = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
                    y_pred_train_actual = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

                    mse_test_actual = mean_absolute_error(y_test_actual, y_pred_test_actual) / np.mean(y_test_actual) * 100
                    mse_train_actual = mean_absolute_error(y_train_actual, y_pred_train_actual) / np.mean(y_train_actual) * 100
                    test_errors_actual.append(mse_test_actual)
                    train_errors_actual.append(mse_train_actual)

                    if epoch%50 == 0:
                            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Error: {mse_train_actual:.4f}, Test Error: {mse_test_actual:.4f}")
            
            else:

                test_errors_actual = []
                train_errors_actual = []

                model_info["n_epochs"] = epochs
                model_info["Batch_size"] = batch_size

                for epoch in tqdm(range(epochs), desc="Training Precess", position=0, leave=True):
                    X_train, y_train = shuffle(X_train, y_train, random_state=42)
                    batch_progress = tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}", position=1, leave=False)

                    for i in batch_progress:
                        X_batch = X_train[i:i+batch_size]
                        y_batch = y_train[i:i+batch_size]
                        model.partial_fit(X_batch, y_batch)
                    
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)
                    
                    y_pred_test_actual = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
                    y_pred_train_actual = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

                    mse_test_actual = mean_absolute_error(y_test_actual, y_pred_test_actual) / np.mean(y_test_actual) * 100
                    mse_train_actual = mean_absolute_error(y_train_actual, y_pred_train_actual) / np.mean(y_train_actual) * 100
                    test_errors_actual.append(mse_test_actual)
                    train_errors_actual.append(mse_train_actual)

                    if epoch%50 == 0:
                        tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Error: {mse_train_actual:.4f}, Test Error: {mse_test_actual:.4f}")
            
            
            if save_model:
                path = self.save_model(path=path, model=model, model_info=model_info, scaler_X=scaler_X, scaler_y=scaler_y, train_error=train_errors_actual, test_error=test_errors_actual, poly=poly)
                self.plot_training_process(path=path, epochs=epochs, test_errors=test_errors_actual, train_errors=train_errors_actual, length=len(X_train), save=True)
                for f in range(X.shape[1]-2):
                    self.plot_model_curve(model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, feature_name=X.columns[f], save_path=path)
            else:
                self.plot_training_process(path=path, epochs=epochs, test_errors=test_errors_actual, train_errors=train_errors_actual, length=len(X_train), save=False)
                # for f in range(X_train.shape[1]-2):
                #     self.plot_model_curve(model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, resolution=50)

    
    def random_forest(self, name, code, n_estimator=100, depth=5, model_save_path=False, return_model=False):
        path = f'/home/hajali/Desktop/Bargh_Ml_project/models/{name}-{code}-randomforest#0'

        X, y = self.extract_data(name, code)

        X_scaled, scaler_X = self.scale(X)
        y_scaled, scaler_y = self.scale(y.values.reshape(-1, 1), do_flat=False)

        X_train, y_train, X_test, y_test = self.split_data(X_scaled, y_scaled)

        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        
        rf_model = RandomForestRegressor(n_estimators=n_estimator, max_depth=depth, random_state=42)

        with tqdm(total=100, desc="Fitting Random forest model.") as pbar:
            rf_model.fit(X_train, y_train)
            for _ in range(10):
                time.sleep(0.1)
                pbar.update(10)

        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)

        # Compute Mean Squared Error (MSE)
        y_pred_test_actual = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
        y_pred_train_actual = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()

        mse_test_actual = mean_absolute_error(y_test_actual, y_pred_test_actual) / np.mean(y_test_actual) * 100
        mse_train_actual = mean_absolute_error(y_train_actual, y_pred_train_actual) / np.mean(y_train_actual) * 100


        print(f"Train Error: {mse_train_actual:.2f}")
        print(f"Test Error: {mse_test_actual:.2f}")

        model_information = {
            "n_estimator": n_estimator,
            "depth": depth,
            "train_mse": mse_train_actual,
            "test_mse": mse_test_actual
        }

        # self.plot_tree(rf_model, X.columns)
        # self.feature_importance(rf_model, X.columns)

        if return_model:
            return rf_model

        if model_save_path:
            path = self.save_model(path=path, model=rf_model, model_info=model_information, scaler_X=scaler_X, scaler_y=scaler_y, train_error=[mse_train_actual], test_error=[mse_test_actual])
            for f in range(X.shape[1]-2):
                self.plot_model_curve(rf_model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, feature_name=X.columns[f], save_path=path)
    
    def xgboost(self, name, code, n_estimator=100, depth=5, epochs=500, learning_rate=0.01, save_model=False):

        path = f'/home/hajali/Desktop/Bargh_Ml_project/models/{name}-{code}-xgboost#0'

        X, y = self.extract_data(name, code)

        X_scaled, scaler_X = self.scale(X)
        y_scaled, scaler_y = self.scale(y.values.reshape(-1, 1), do_flat=True)

        X_train, y_train, X_test, y_test = self.split_data(X_scaled, y_scaled)

        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'reg:squarederror',
            'learning_rate': learning_rate,
            'max_depth': depth,
            'eval_metric': 'mae'
        }

        num_boost_rounds = epochs
        evals_result = {}

        pbar = tqdm(total=num_boost_rounds, desc="Training Progress", unit="epoch")

        mae_callback = MAECallback(dtrain, dtest, y_train_actual, y_test_actual, scaler_y)

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_rounds,
            evals=[(dtrain, "Train"), (dtest, "Test")],
            evals_result=evals_result,
            verbose_eval=False,
            callbacks=[TQDMCallback(total_epochs=epochs), mae_callback]
        )

        pbar.close()

        train_mae_per_epoch = mae_callback.train_mae_per_epoch
        test_mae_per_epoch = mae_callback.test_mae_per_epoch

        # Optionally convert to percentage error
        train_mae_percentage = np.array(train_mae_per_epoch) / np.mean(y_train_actual) * 100
        test_mae_percentage = np.array(test_mae_per_epoch) / np.mean(y_test_actual) * 100

        print(f"Train Error: {np.mean(train_mae_percentage)}")
        print(f"Test Error: {np.mean(test_mae_percentage)}")

        model_information = {
            'objective': 'reg:squarederror',
            'learning_rate': learning_rate,
            'max_depth': depth,
            'eval_metric': 'rmse',
            "n_epochs": epochs
        }

        if save_model:
            path = self.save_model(path=path, model=bst, model_info=model_information, scaler_X=scaler_X, scaler_y=scaler_y, train_error=train_mae_percentage, test_error=test_mae_percentage)
            self.plot_training_process(path=path, epochs=epochs, test_errors=test_mae_percentage, train_errors=train_mae_percentage, length=len(X_train), save=save_model)
            for f in range(X.shape[1] - 2):
                self.plot_model_curve(bst, X=X_train, y=y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, feature_name=X.columns[f], save_path=path)
        
        else:
            self.plot_training_process(path=path, epochs=epochs, test_errors=test_mae_percentage, train_errors=train_mae_percentage, length=len(X_train), save=False)
            # for f in range(X_train.shape[1]-2):
            #     self.plot_model_curve(bst, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, feature_name=X.columns[f])
        
    
    def svr(self, name, code, kernel: str, C=1, epsilon=0.1, gamma='scale', save_model=False):
        df_copy = self.df[(self.df['name'] == name) & (self.df['code'] == code)]
        path = f'/home/hajali/Desktop/Bargh_Ml_project/models/{name}-{code}-svr#0'

        if len(df_copy.index) < 2:
            print(f'number of elements less than 2')
            return None
        
        df = df.select_dtypes(include=np.number)

        X = df.drop(columns=[self.target])
        y = df[self.target]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.values.reshape(-1,1)).flatten()

        if len(X) < 5:
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

        with tqdm(total=100, desc="Fitting Random forest model.") as pbar:
            svr_model.fit(X_train, y_train)
            for _ in range(10):
                time.sleep(0.1)
                pbar.update(10)

        y_train_pred = svr_model.predict(X_train)
        y_test_pred = svr_model.predict(X_test)

        # Compute Mean Squared Error (MSE)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")

        if save_model:

            counter = 1

            while os.path.exists(path=path):
                path = f"{path.split('#')[0]}{counter}"
                counter += 1
            
            os.mkdir(path=path)

            joblib.dump(svr_model, f"{path}/model.pkl")
            joblib.dump(scaler_y, f"{path}/scaler_y.pkl")
            joblib.dump(scaler_X, f"{path}/scaler_X.pkl")
            evaluations = {
                'train_MSE': train_mse,
                'test MSE': test_mse
            }
            joblib.dump(evaluations, f"{path}/eval.pkl")
    
    def kernel_ridge(self, name, code, kernel: str, alpha, gamma, degree=None, coef=None, save_model=False):
        path = f'/home/hajali/Desktop/Bargh_Ml_project/models/{name}-{code}-kernel_ridge#0'

        X, y = self.extract_data(name, code)

        X_scaled, scaler_X = self.scale(X)
        y_scaled, scaler_y = self.scale(y.values.reshape(-1, 1), do_flat=True)

        X_train, y_train, X_test, y_test = self.split_data(X_scaled, y_scaled)

        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()

        kr_model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma) if degree==None else KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef)

        # kr_model = make_pipeline(
        #     RBFSampler(gamma=gamma, n_components=5000, random_state=42),
        #     Ridge(alpha=alpha)
        # )

        kr_model.fit(X_train, y_train)
        print("model fitted")

        y_train_pred = kr_model.predict(X_train)
        y_test_pred = kr_model.predict(X_test)

        print("prediction acomplished.")

        y_pred_test_actual = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
        y_pred_train_actual = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()

        mse_test_actual = mean_absolute_error(y_test_actual, y_pred_test_actual) / np.mean(y_test_actual) * 100
        mse_train_actual = mean_absolute_error(y_train_actual, y_pred_train_actual) / np.mean(y_train_actual) * 100

        model_information = {
            "kernel": kernel,
            "gamma": gamma,
            "alpha": alpha,
            "degree": degree,
            "coef0": coef
        }

        if save_model:
            path = self.save_model(path=path, model=kr_model, model_info=model_information, scaler_X=scaler_X, scaler_y=scaler_y, train_error=[mse_train_actual], test_error=[mse_test_actual])
            for f in range(X.shape[1]-2):
                self.plot_model_curve(kr_model, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, feature_index=f, feature_name=X.columns[f], save_path=path)
        else:
            print(f"Train Error: {mse_train_actual}")
            print(f"Test Error: {mse_test_actual}")
    
    def plot_model_curve(self, model, X, y, scaler_X, scaler_y, resolution=100, feature_index=0, feature_name=None, save_path=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[1] < 1:
            print(f"Multiple feature detected! using feature index {feature_index} for plotting.")
        
        X_plot = X[:, feature_index]

        X_range = np.linspace(X_plot.min(), X_plot.max(), resolution).reshape(-1, 1)

        X_range_full = np.zeros((resolution, X.shape[1]))
        X_range_full[:, feature_index] = X_range[:, 0]
        
        if hasattr(model, "named_steps") and "polynomialfeatures" in model.named_steps:
            poly_transformer = model.named_steps["polynomialfeatures"]
            X_range_transformed = poly_transformer.transform(X_range_full)
            y_pred = model.predict(X_range_transformed)

        # Handle XGBoost models
        elif isinstance(model, xgb.Booster):
            dtest = xgb.DMatrix(X_range_full)
            y_pred = model.predict(dtest)

        # Handle all other models (Linear, SGD, SVR, etc.)
        else:
            y_pred = model.predict(X_range_full)
        
        X_plot_original = scaler_X.inverse_transform(X)[:, feature_index]
        X_range_original = scaler_X.inverse_transform(X_range_full)[:, feature_index]
        y_original = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        plt.scatter(X_plot_original, y_original, color='blue', label='Actual data', alpha=0.5)
        plt.plot(X_range_original, y_pred_original, color='red', linewidth=2, label='Model prediction')

        plt.xlabel(f"Feature {feature_name}")
        plt.ylabel("Target value")
        plt.title("Model curve on Data")
        plt.legend()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(f"{save_path}/{feature_name}_vs_target.jpg")
            plt.close()
    
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
        
        return (X_train, y_train, X_test, y_test)
    
    def create_sgd(self, penalty=None, max_iter=1, tol=1e-3, alpha=0.01, lr="constant", initial_lr=0.01, warm_start=True):

        model = SGDRegressor(penalty=penalty, max_iter=max_iter, tol=tol, alpha=alpha, learning_rate=lr, eta0=initial_lr, warm_start=warm_start)

        model_info = model.get_params().items()

        logger.info(
                msg="Model created.\nModel configuration: SGDRegressor(" + ", ".join(f"{key}={value}" for key, value in model_info) + ")"
            )

        return model, dict(model_info)
    
    def save_model(self, path, model, model_info, scaler_X, scaler_y, train_error, test_error, poly=None):

        try:
            counter = 1

            while os.path.exists(path=path):
                path = f"{path.split('#')[0]}#{counter}"
                counter += 1
            
            os.mkdir(path=path)
            
            joblib.dump(model, f"{path}/model.pkl")
            logger.info(msg=f"Model saved in path: {path}/model.pkl")

            joblib.dump(scaler_y, f"{path}/scaler_y.pkl")
            logger.info(msg=f"Scaler of Model saved in path: {path}/scaler_y.pkl")

            joblib.dump(scaler_X, f"{path}/scaler_X.pkl")
            logger.info(msg=f"Scaler of Model saved in path: {path}/Scaler_X.pkl")

            eval_results = {
                'Train': train_error,
                'Test': test_error
            }
            joblib.dump(eval_results, f"{path}/eval.pkl")
            logger.info(msg=f"Evaluations of the model saved in path: {path}/eval.pkl")
            
            model_info['droped_features'] = self.droped
            with open(f"{path}/model_info.json", 'w') as f:
                json.dump(model_info, f, indent=4)
            
            logger.info(msg=f"Model information saved in the path: {path}/model_info.json")

            if poly is not None:
                joblib.dump(poly, f"{path}/poly.pkl")
                logger.info(msg=f"Poly transformation saved in the path: {path}/poly.pkl")

            id = path.split('/')[-1].split('.')[0]

            bnchmrk.add_model(train_error, test_error, id, N=10, is_generation_predictor=True)

            return path
        except Exception as e:
            logger.error(msg=f"Could not save the model and parameters of the model. Exception below occured:\n{e}")
    
    def plot_training_process(self, path, epochs, test_errors, train_errors, length, save=False):
        try:
            plt.figure(figsize=(12, 8))
            plt.plot(range(1, epochs+1), test_errors, label='Test MSE', color='blue')
            plt.plot(range(1, epochs+1), train_errors, label='Train MSE', color='red')
            plt.xlabel('epochs')
            plt.ylabel('Mean Squared Erroe')
            plt.title('Training Process')
            plt.legend()
            plt.grid()
            plt.text(x=0.5, y=0.9, s=f'# of data: {length}', fontsize=12, ha='center', va='top', transform=plt.gca().transAxes)
            
            if save:
                plt.savefig(f"{path}/TrainingProcess.jpg")
                plt.close()
                logger.info(msg=f"Figure of training process of the model saved in the path: {path}/TrainingProcess.png")
            else:
                plt.show()
        except Exception as e:
            logger.error(msg=f"Could not plot the training process. Exception below occured:\n{e}")
    
    def extract_data(self, name, code):
        if code is not None:
            df_copy = self.df[(self.df['name'] == name) & (self.df['code'] == code)]
        else:
            df_copy = self.df[self.df['name'] == name]
        print(f"number of data is: {len(df_copy)}")
        logger.info(msg=f"Successfully splitted the {name}-{code} from the original dataset to train on.")

        if len(df_copy.index) < 2:
            logger.warning(msg=f"Length of the dataset is less than 2 and training stopped on it. returning None.")
            return None, None
        
        df_copy.drop(columns=['id', 'hour', 'date', 'name', 'code', 'status', 'declare'], axis=1, inplace=True)

        null_columns = df_copy.columns[df_copy.isnull().all()].tolist()
        df_copy = df_copy.drop(columns=null_columns)
        seasonality_cols_base = ['year_effect', 'week_effect', 'hour_effect']
        seasonality_cols = [x for x in seasonality_cols_base if not x in null_columns]

        df_copy = df_copy.dropna()

        df_copy = pd.get_dummies(df_copy, columns=seasonality_cols)

        print(f"Features training is applied on: {df_copy.columns}")

        # df_copy = self.remove_outliers(df=df_copy)

        X = df_copy.drop(columns=[self.target])
        y = df_copy[self.target]

        return X, y
    
    def evaluate_prcntg(self, y_train, y_test, y_pred_train, y_pred_test):
        pass
    @classmethod
    def predict(self, input, model_path, is_poly=False):
        if is_poly:
            model, scaler_X, scaler_y, model_info, poly = self.load_model(model_path=model_path, is_poly=True)
        else:
            model, scaler_X, scaler_y, model_info = self.load_model(model_path=model_path)
        input = input.drop(columns=['id', 'hour', 'date', 'name', 'code', 'status', 'declare'])
        input = input.drop(columns=model_info['droped_features'])
        x = input.drop(columns=['generation'])
        x = poly.transform(x) if is_poly is True else x
        x_scaled = scaler_X.transform(x)
        if type(model) == xgb.core.Booster:
            x_scaled = xgb.DMatrix(x_scaled)
        y_pred_scaled = (model.predict(x_scaled)).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        return y_pred

    @classmethod
    def load_model(self, model_path, is_poly=False):
        with open(f"{model_path}/model.pkl", 'rb') as f:
            model = joblib.load(f)
        
        with open(f"{model_path}/scaler_X.pkl", 'rb') as f:
            scaler_X = joblib.load(f)
        
        with open(f"{model_path}/scaler_y.pkl", 'rb') as f:
            scaler_y = joblib.load(f)
        
        with open(f"{model_path}/model_info.json", 'rb') as f:
            model_info = json.load(f)
        
        if is_poly:
            with open(f"{model_path}/poly.pkl", 'rb') as f:
                poly = joblib.load(f)
            return (model, scaler_X, scaler_y, model_info, poly)
        else:
            return (model, scaler_X, scaler_y, model_info)
    
    def plot_tree(self, model, feature_names):
        tree = model.estimators_[0]

        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            special_characters=True
        )

        graph = graphviz.Source(dot_data)
        graph.render("regression tree", format="png", cleanup=False)
        graph.view()
    
    def feature_importance(self, model, feature_names):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        sorted_feature_names = np.array(feature_names)[indices]
        sorted_importances = importances[indices]

        plt.barh(sorted_feature_names, sorted_importances)
        plt.xlabel("Feature importance")
        plt.title("Overall feature importance in random forest")
        plt.show()
    
    def remove_outliers(self, df:pd.DataFrame):
        l = len(df.index)
        columns = df.select_dtypes(include='number').columns.tolist()
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        print(f"{l - len(df.index)} number of data were outlier and removed.")
        return df
    
    @classmethod
    def evaluate_unit(self, input: pd.DataFrame, predictions):
        # sample = self.df[(self.df['name'] == name) & (self.df['code'] == code)]
        sample = input
        error = mean_absolute_error(sample['generation'], predictions)
        error_percentage = error / np.mean(sample['generation']) * 100
        return error_percentage
    
    def plot_prediction(self, name, code, predictions, save=False):
        sample = self.df[self.df['name'] == name]

        if code is not None:
            sample = sample[sample['code'] == code]
        
        # sample['color'] = np.where(sample['value_P'] == True, 'red', np.where(sample['value_M'] == True, 'blue', 'green'))
        sample['prediction'] = predictions

        sample['date'] = pd.to_datetime(sample['date'])
        sample['datetime'] = sample['date'] + pd.to_timedelta(sample['hour'], unit='H')
        sample['datetime'] = sample['datetime'].astype('datetime64[ns]')
        sample.sort_values(by='datetime')

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sample['datetime'],
            y=sample['generation'],
            mode="lines",
            name="Actual",
            line=dict(color="blue", dash="solid"),
        ))

        fig.add_trace(go.Scatter(
            x=sample['datetime'],
            y=sample['generation'],
            mode='markers',
            marker=dict(color="blue"),
            name="Actual"
        ))

        fig.add_trace(go.Scatter(
            x=sample['datetime'],
            y=sample['prediction'],
            mode="lines",
            name=f"Prediction",
            line=dict(color="yellow", dash="solid"),
        ))

        fig.show() if not save else fig.write_html(f"/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/Dividing/{name}-{code}-original.html")
