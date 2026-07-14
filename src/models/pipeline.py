import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from classifier import Classifer
from best_model import BestModel
from true_generation_model import Model
import csv
import numpy as np

class Pipeline:

    def __init__(self, df: pd.DataFrame, features_to_drop=None):
        # if features_to_drop is not None:
        #     self.df = df.drop(columns=features_to_drop)
        # else:
        #     self.df = df
        self.df = df
        
        self.droped = features_to_drop
        self.best_regressors = "/home/hajali/Desktop/Bargh_Ml_project/models/best_models.json"
        self.best_classifiers = "/home/hajali/Desktop/Bargh_Ml_project/models/best_classifiers.json"
        self.benchmark = "/home/hajali/Desktop/Bargh_Ml_project/models/benchmark/GenerationPredictors.csv"
    
    def execute(self, name, code, save=False):
        sample = self.df[(self.df['name'] == name) & (self.df['code'] == code)]

        id = "-".join([
            str(sample.iloc[0]['name']),
            str(sample.iloc[0]['code'])
        ])

        id2 = "-".join([
            str(sample.iloc[0]['name']),
            str(sample.iloc[0]['code'])
        ])

        with open(self.best_classifiers, 'r') as f:
            classifiers_file = json.load(f)
        
        model_name = classifiers_file[id]

        with open(self.best_regressors, 'r') as f:
            regressor_files = json.load(f)
        
        # regressor_name = regressor_files[id2]

        model_path = f"/home/hajali/Desktop/Bargh_Ml_project/models/classifiers/{model_name}"

        print(f"$$$$$$$$$$$$$$$$$ classifier model name: {model_name}")

        sample = Classifer.predict(input=sample, model_path=model_path, droped_features=self.droped, return_df=True)

        unique_labels = sample['label'].unique()

        # original_label = sample['label'].mode()[0]

        print(f"%%%%%%%%%%%%%%%%%%%%%%% labels: {unique_labels}")

        dfs = {val: sample[sample['label'] == val] for val in unique_labels}

        bm = BestModel()

        united_data = pd.DataFrame(columns=['date', 'hour', 'actual', 'prediction', 'label'])

        l = len(sample.index)
        final_error = 0

        for key in dfs.keys():
            # if key == original_label:
            #     input = dfs[key]

            #     y_actual = input['generation']
            #     y_pred = bm.best_prediction(input=input.drop(columns=['label']))

            #     temp_df = pd.DataFrame()
            #     temp_df['date'] = input['date']
            #     temp_df['hour'] = input['hour']
            #     temp_df['actual'] = y_actual
            #     temp_df['prediction'] = y_pred
            #     temp_df['label'] = key

            #     united_data = pd.concat([united_data, temp_df])

            #     final_error += (mean_absolute_error(y_actual, y_pred) * (len(y_actual) / l)) / np.mean(y_actual) * 100
            
            input = dfs[key]
            print(input.head())

            model = RandomForestRegressor(n_estimators=150, max_depth=80, random_state=42)
            x_train, x_test, y_train, y_test = train_test_split(input[['temperature', 'value_P', 'value_M', 'surface_pressure', 'humidity']], input['generation'], test_size=0.2, random_state=42)
            model.fit(x_train, y_train)

            y_actual = input['generation']
            y_pred = model.predict(input[['temperature', 'value_P', 'value_M', 'surface_pressure', 'humidity']])

            temp_df = pd.DataFrame()
            temp_df['date'] = input['date']
            temp_df['hour'] = input['hour']
            temp_df['actual'] = y_actual
            temp_df['prediction'] = y_pred
            temp_df['label'] = key

            united_data = pd.concat([united_data, temp_df])

            final_error += (mean_absolute_error(y_actual, y_pred) * (len(y_actual) / l)) / np.mean(y_actual) * 100


        
        united_data['date'] = pd.to_datetime(united_data['date'])
        united_data['datetime'] = united_data['date'] + pd.to_timedelta(united_data['hour'], unit='H')
        united_data['datetime'] = united_data['datetime'].astype('datetime64[ns]')

        color_map = px.colors.qualitative.Dark24
        color_dict = {label: color_map[i % len(color_map)] for i, label in enumerate(unique_labels)}

        fig = go.Figure()

        for label in unique_labels:
            df_label = united_data[united_data['label'] == label]

            fig.add_trace(go.Scatter(
                x=df_label['datetime'],
                y=df_label['actual'],
                mode="lines",
                name=f"{label} Actual",
                line=dict(color=color_dict[label], dash="solid"),
                legendgroup=str(label),
                hovertemplate=f"Label: {label}<br>Actual Generation: %{{y}}<br>Time: %{{x}}<extra></extra>"
            ))

            fig.add_trace(go.Scatter(
                x=df_label['datetime'],
                y=df_label['actual'],
                mode='markers',
                marker=dict(color=color_dict[label]),
                name="Actual"
            ))

            fig.add_trace(go.Scatter(
                x=df_label['datetime'],
                y=df_label['prediction'],
                mode="lines",
                name=f"{label} Prediction",
                line=dict(color="yellow", dash="solid"),
                legendgroup=str(label),
                hovertemplate=f"Label: {label}<br>Predicted Generation: %{{y}}<br>Time: %{{x}}<extra></extra>"
            ))
        
        fig.update_layout(
        title='Actual vs Predicted Prices per Label',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified'
        )

        fig.show() if not save else fig.write_html(f"/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/Dividing/{name}-{code}-divided.html")

        bench = pd.read_csv(self.benchmark)
        # prv_error = float(bench[bench['id'] == regressor_name]['last test'].values)

        with open("/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/Dividing/result.csv", 'a') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow([f"{name}-{code}", np.round(final_error, 4), np.round(prv_error, 4)])
            writer.writerow([f"{name}-{code}", np.round(final_error, 4)])
    
    def by_bar(self, name, code, save=False):
        sample = self.df[(self.df['name'] == name) & (self.df['code'] == code)]

        df_P = sample[sample['value_P'] == True]
        df_M = sample[sample['value_M'] == True]
        df_L = sample[(sample['value_P'] == False) & (sample['value_M'] == False)]

        united_data = pd.DataFrame(columns=['date', 'hour', 'actual', 'prediction', 'label'])

        l = len(sample.index)

        dfs = {
            0: df_L,
            1: df_M,
            2: df_P
        }

        final_error = 0

        for val in dfs.keys():
            
            df = dfs[val]

            model = RandomForestRegressor(n_estimators=150, max_depth=80, random_state=42)
            x_train, x_test, y_train, y_test = train_test_split(df[['temperature', 'surface_pressure', 'humidity']], df['generation'], test_size=0.2, random_state=42)
            model.fit(x_train, y_train)

            y_actual = df['generation']
            y_pred = model.predict(df[['temperature', 'surface_pressure', 'humidity']])

            temp_df = pd.DataFrame()
            temp_df['date'] = df['date']
            temp_df['hour'] = df['hour']
            temp_df['actual'] = y_actual
            temp_df['prediction'] = y_pred
            temp_df['label'] = val

            united_data = pd.concat([united_data, temp_df])

            final_error += (mean_absolute_error(y_actual, y_pred) * (len(y_actual) / l)) / np.mean(y_actual) * 100
        
        united_data['date'] = pd.to_datetime(united_data['date'])
        united_data['datetime'] = united_data['date'] + pd.to_timedelta(united_data['hour'], unit='H')
        united_data['datetime'] = united_data['datetime'].astype('datetime64[ns]')

        color_map = px.colors.qualitative.Dark24
        color_dict = {label: color_map[i % len(color_map)] for i, label in enumerate([0, 1, 2])}

        unique_labels = [0, 1, 2]

        bar = {
            0: "Low",
            1: "Medium",
            2: "Peak"
        }

        fig = go.Figure()

        for label in unique_labels:
            df_label = united_data[united_data['label'] == label]

            fig.add_trace(go.Scatter(
                x=df_label['datetime'],
                y=df_label['actual'],
                mode="lines",
                name=f"{label} Actual",
                line=dict(color=color_dict[label], dash="solid"),
                legendgroup=str(label),
                hovertemplate=f"Label: {label}<br>Actual Generation: %{{y}}<br>Time: %{{x}}<extra></extra>"
            ))

            fig.add_trace(go.Scatter(
                x=df_label['datetime'],
                y=df_label['actual'],
                mode='markers',
                marker=dict(color=color_dict[label]),
                name=bar[label]
            ))

            fig.add_trace(go.Scatter(
                x=df_label['datetime'],
                y=df_label['prediction'],
                mode="lines",
                name=f"{label} Prediction",
                line=dict(color="yellow", dash="solid"),
                legendgroup=str(label),
                hovertemplate=f"Label: {label}<br>Predicted Generation: %{{y}}<br>Time: %{{x}}<extra></extra>"
            ))
        
        fig.update_layout(
        title=f'Actual vs Predicted Generation for {name}-{code}',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified'
        )

        fig.show() if not save else fig.write_html(f"/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/Dividing/{name}-{code}-bar.html")

        id2 = "-".join([
            str(sample.iloc[0]['name']),
            str(sample.iloc[0]['code'])
        ])

        with open(self.best_regressors, 'r') as f:
            regressor_files = json.load(f)
        
        regressor_name = regressor_files[id2]

        bench = pd.read_csv(self.benchmark)
        prv_error = float(bench[bench['id'] == regressor_name]['last test'].values)

        with open("/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/Dividing/result.csv", 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"{name}-{code}-bar", np.round(final_error, 4), np.round(prv_error, 4)])