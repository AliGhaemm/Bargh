import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
import plotly.express as px
import plotly.graph_objects as go

logger = CustomLogger(name="Cluster", log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/Clustering.log').get_logger()

class Cluster:

    def __init__(self, base_path=None):
        self.base_path = base_path

    def mean_cluster(self, data):

        threshold = data['generation'].mean()
        data['color'] = np.where(data['generation'] >= threshold, 'red', np.where(data['generation'] < threshold, 'blue', 'green'))
        self.plot_generation(sample=data, threshold=threshold)
        

    def kmeans_cluster(self, data, n=2, on_generation=False, labeling=False, save=False):
        x = data.drop(columns=['generation']) if not on_generation else data['generation'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n, random_state=0).fit(x)
        labels = kmeans.labels_
        data['group'] = labels
        data['color'] = np.where(data['group'] == 1, 'red', np.where(data['group'] == 0, 'blue', 'green'))
        centers = np.sort(kmeans.cluster_centers_.ravel())
        seperator = (centers[0] + centers[1]) / 2
        boundries = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

        if labeling:
            # self.interactive_plot(sample=data, save=save, n=n)
            return data.drop(columns=['color'])
        
        self.interactive_plot(sample=data, save=save, n=n)

    def gaussian_cluster(self, data, n=2, on_generation=False, labeling=False, save=False):
        x = data.drop(columns=['generation']) if not on_generation else data['generation'].values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n, random_state=0).fit(x)
        labels = gmm.predict(x)
        data['group'] = labels
        data['color'] = np.where(data['group'] == 1, 'red', np.where(data['group'] == 0, 'blue', 'green'))
        if labeling:
            # self.interactive_plot(sample=data, save=save, n=n, is_g=True)
            return data.drop(columns=['color'])

        self.interactive_plot(sample=data, save=save, n=n, is_g=True)
    
    def hmm(self, sample):
        sample['date'] = pd.to_datetime(sample['date'])
        sample['returns'] = sample['generation'].diff().fillna(0)
        X = sample[['returns']].values
        hmm = GaussianHMM(n_components=2, covariance_type='full', n_iter=1000, random_state=42)
        hmm.fit(X)

        hidden_states = hmm.predict(X)
        sample['state'] = hidden_states

        plt.figure(figsize=(48, 24))

        for i in range(hmm.n_components):
            state_mask = (hidden_states == i)
            plt.plot(sample['date'][state_mask], sample['generation'][state_mask], '.', label=f"State {i}")
        
        plt.legend()
        plt.title('Generation Colored by Hidden Markov States')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    def plot_generation(self, sample, threshold=None):
        sample['date'] = pd.to_datetime(sample['date'])
        # sample['datetime'] = sample['date'] + pd.to_timedelta(sample['hour'], unit='H')
        # sample['datetime'] = sample['datetime'].astype('datetime64[ns]')
        # sample.sort_values(by='datetime')
        # sample.set_index('datetime')

        plt.figure(figsize=(48, 24))

        plt.plot(sample['date'], sample['generation'], label='Actual generation', color='blue', linewidth=3)
        plt.scatter(sample['date'], sample['generation'], c=sample['color'], s=60)
        if threshold is not None:
            if not isinstance(threshold, np.ndarray) and not isinstance(threshold, list):
                plt.axhline(y=threshold, color='black', linestyle='--', linewidth=3)
            else:
                print("inside for loop")
                for b in threshold:
                    plt.axhline(y=b, color='black', linestyle='--', linewidth=3)
        plt.xlabel('Time', fontsize=35)
        plt.ylabel('generation', fontsize=35)
        plt.legend(fontsize=30)
        name = sample['name'].iloc[0]
        code = sample['code'].iloc[0]
        title = f"{name}-{code}"
        plt.title(title, fontsize=60)
        plt.xticks(rotation=90, fontsize=30)
        plt.yticks(fontsize=30)
        plt.grid(True)
        # plt.savefig(f"/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/Clustering/Gaussian/{name}-{code}-gaussian.png")
        plt.show()
    

    def interactive_plot(self, sample, threshold=None, save=False, n=2, is_g=False, is_hmm=False):

        sample['date'] = pd.to_datetime(sample['date'])
        sample['datetime'] = sample['date'] + pd.to_timedelta(sample['hour'], unit='H')
        sample['datetime'] = sample['datetime'].astype('datetime64[ns]')

        title = f"{sample['name'].iloc[0]}-{sample['code'].iloc[0]}-{n}"

        if is_g:
            save_path = f"{self.base_path}/GaussianMixture/{title}"
        elif is_hmm:
            save_path = f"{self.base_path}/HMM/{title}"
        else:   
            save_path = f"{self.base_path}/Kmeans/{title}"
        
        fig1 = px.scatter(sample, x='datetime', y='generation', color='color', title='Generation clusters')

        fig1.update_layout(
            xaxis_title='Time',
            yaxis_title='Generation',
            xaxis=dict(type='date')
        )

        fig1.show() if not save else fig1.write_html(f"{save_path}-scatter.html")

        scatter_trace = go.Scatter(
            x = sample['datetime'],
            y = sample['generation'],
            mode='markers',
            marker=dict(color=sample['color']),
            name='Generation'
        )

        line_trace = go.Scatter(
            x = sample['datetime'],
            y = sample['generation'],
            mode = 'lines',
            line=dict(color='black'),
            name='Line Plot'
        )

        fig2 = go.Figure(data=[scatter_trace, line_trace])

        fig2.update_layout(
            title="Generation clusters",
            xaxis_title='Time',
            yaxis_title='Generation',
            xaxis=dict(type='date'),
            template='plotly_dark'
        )

        fig2.show() if not save else fig2.write_html(f"{save_path}-line.html")