import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.cm as cm
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import convertdate
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import plotly.graph_objects as go
from datetime import timedelta


class UnitPlotter:

    def __init__(self):
        if not os.path.exists('/home/hajali/Desktop/Bargh_Ml_project/data/processed/operation_forplot.csv'):
            file_path = '/home/hajali/Desktop/Bargh_Ml_project/data/processed/operation.csv'
            df = pd.read_csv(file_path)
            self.df = self.assign_color(df=df)
            # self.df['date'] = self.df['date'].apply(self.jalali_to_gregorian)
            self.df['date'] = pd.to_datetime(self.df['date'])

            self.df['datetime'] = self.df['date'] + pd.to_timedelta(self.df['hour'], unit='H')
            self.df.to_csv('/home/hajali/Desktop/Bargh_Ml_project/data/processed/operation_forplot.csv')
        else:
            self.df = pd.read_csv('/home/hajali/Desktop/Bargh_Ml_project/data/processed/operation_forplot.csv')
            self.df['color'] = self.df['color'].apply(self.convert_to_tuple)
            self.df['datetime'] = self.df['datetime'].astype('datetime64[ns]')
            self.df['date'] = pd.to_datetime(self.df['date'])
    
    def convert_to_tuple(self, x: str):
        elements = x.strip('()').split(', ')

        converted_elements = []
        for element in elements:
            try:
                converted_elements.append(int(element))
            except ValueError:
                try:
                    converted_elements.append(float(element))
                except ValueError:
                    converted_elements.append(element.strip("'"))
        
        return tuple(converted_elements)

    def hsv_to_rgb(self, h, s, v):
        if s == 0.0:
            return(v, v, v)
        
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i %= 6
        return {
            0: (v, t, p),
            1: (q, v, p),
            2: (p, v, t),
            3: (p, q, v),
            4: (t, p, v),
            5: (v, p, q),
        }[i]


    def golden_ration(self, n):
        colors = []
        phi = (1 + 5 ** 0.5) / 2
        for i in range(n):
            hue = (i / phi) % 1
            color = self.hsv_to_rgb(hue, 1, 1)
            colors.append(color)
        return colors
    
    def assign_color(self, df: pd.DataFrame):
        statuses = df['status'].unique()
        status_count = len(statuses)
        colors = self.golden_ration(status_count)
        color_col = {
            stat: c for stat, c in zip(statuses, colors)
        }
        colorized_df = df.copy()
        colorized_df['color'] = colorized_df['status'].map(color_col)
        return colorized_df
    
    def jalali_to_gregorian(self, date_str):
        try:
            year, month, day = map(int, date_str.split('/'))
            g_year, g_month, g_day = convertdate.persian.to_gregorian(year, month, day)
            return f'{g_year}-{g_month:02d}-{g_day:02d}'
        except Exception as e:
            print(f'Exception occured: {e}')
    
    def plot_generation_vs_temp(self, name, code, save=False):
        self.simple_plot(name=name, code=code, x_name='temperature', y_name='generation',
                         save_path=f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/GenerationVStemperature/{name}-{code}-genVStemp-sctr.png')
    
    def plot_declaration_vs_temp(self, name, code, save=False):
        self.simple_plot(name=name, code=code, x_name="temperature", y_name="declare",
                         save_path=f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/DeclarationVStemperature/{name}-{code}-decVStemp-sctr.png')
        
    def plot_generation_over_time(self, name, code, save=False):
        self.simple_plot(name=name, code=code, x_name='datetime', y_name='generation',
                         save_path=f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/GenerationOverTime/{name}-{code}-GenOverTime-plt.png')
    
    def plot_declaration_over_time(self, name: str, code, save=False):
        self.simple_plot(name=name, code=code, x_name='datetime', y_name='declare',
                         save_path=f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/DeclarationOverTime/{name}-{code}-DeclOverTime-plt.png')
    
    def plot_status_over_time(self, name, code, save=False):
        sample = self.df.loc[(self.df['name'] == name) & (self.df['code'] == code)]
        sample = sample.sort_values(by='datetime')
        sample = sample.set_index('datetime')

        status_mapping = {status: i for i, status in enumerate(set(sample['status']))}
        sample['status_num'] = sample['status'].map(status_mapping)

        plt.figure(figsize=(24, 12))
        plt.plot(sample.index, sample['status_num'], marker='o')

        plt.xticks(rotation=90)
        plt.yticks(range(len(status_mapping)), status_mapping.keys())

        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Status', fontsize=20)
        plt.title('Status Over Time', fontsize=30)

        if not save:
            plt.show()
        else:
            plt.savefig(f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/StatusOverTime/{name}-{code}-statusOverTime-plt.png')
            plt.close()
    
    def plot_generation_and_temp_over_time(self, name, code, save=False):
        sample = self.df.loc[(self.df['name'] == name) & (self.df['code'] == code)]
        sample = sample.sort_values(by='datetime')
        sample = sample.set_index('datetime')

        fig, ax1 = plt.subplots(figsize=(24, 12))

        color = 'tab:red'
        ax1.set_xlabel('Date', fontsize=40)
        plt.xticks(rotation=90)
        ax1.set_ylabel('Generation', color=color, fontsize=40)
        ax1.plot(sample.index, sample['generation'], color=color, linewidth=3)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Temperature', color=color, fontsize=40)
        ax2.plot(sample.index, sample['temperature'], color=color, linewidth=3)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()

        if not save:
            plt.show()
        else:
            plt.savefig(f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/GenerationTemperatureOverTime/{name}-{code}-GenTempOverTime-plt.png')
            plt.close()
    
    def plot_generation_and_declare_over_time(self, name, code, save=False):
        sample = self.df.loc[(self.df['name'] == name) & (self.df['code'] == code)]
        sample = sample.sort_values(by='datetime')
        sample = sample.set_index('datetime')

        # fig, ax1 = plt.subplots(figsize=(24, 12))

        # color = 'tab:red'
        # ax1.set_xlabel('Date', fontsize=40)
        # plt.xticks(rotation=90)
        # ax1.set_ylabel('Generation', color=color, fontsize=40)
        # ax1.plot(sample.index, sample['generation'], color=color, linewidth=3)
        # ax1.tick_params(axis='y', labelcolor=color)

        # ax2 = ax1.twinx()

        # color = 'tab:blue'
        # ax2.set_ylabel('Declare', color=color, fontsize=40)
        # ax2.plot(sample.index, sample['declare'], color=color, linewidth=3)
        # ax2.tick_params(axis='y', labelcolor=color)

        # plt.grid(True)
        # fig.tight_layout()

        plt.figure(figsize=(12, 6))
        plt.plot(sample.index, sample['generation'], color='tab:red', linewidth=3, label='Generation')
        plt.plot(sample.index, sample['declare'], color='tab:blue', linewidth=3, label='Declaration')
        plt.xlabel('Time')
        plt.ylabel('Generation and Declaration')
        plt.legend()
        plt.xticks(rotation=90)

        if not save:
            plt.show()
        else:
            plt.savefig(f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/GenerationDeclarationOverTime/{name}-{code}-GenDeclOverTime-plt.png')
            plt.close()
    
    def plot_temperature_over_time(self, name, code, save=False):
        temp = pd.read_csv("/home/hajali/Desktop/Bargh_Ml_project/data/interim/")
        sample = self.df.loc[(self.df['name'] == name) & (self.df['code'] == code)]
        sample = sample.sort_values(by='datetime')
        sample = sample.set_index('datetime')

        plt.figure(figsize=(12, 6))
        plt.plot(sample.index, sample['temperature'], color='black', linewidth=2, label='Temperature')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.legend()
        plt.xticks(rotation=90)

        if not save:
            plt.show()
        else:
            plt.savefig(f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/GenerationDeclarationOverTime/{name}-{code}-TempOverTime-plt.png')
            plt.close()
    
    def plot_operation_features(self, name, code, save=False):
        sample = self.df.loc[(self.df['name'] == name) & (self.df['code'] == code)]
        sample = sample.sort_values(by='datetime')
        sample.set_index('datetime', inplace=True)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        sample = sample.select_dtypes(include=numerics)
        sample.drop(columns=['hour', 'unitid'], inplace=True)

        cmap = cm.get_cmap('viridis')
        colors = [cmap(i / (len(sample.columns) - 2)) for i in range(len(sample.columns))]

        fig, axs1 = plt.subplots(figsize=(48, 24))

        axs1.plot(sample.index, sample[sample.columns[0]], color=colors[0])
        axs1.set_xlabel('Time')
        axs1.set_ylabel(sample.columns[0], color=colors[0])
        axs1.tick_params(axis='y', labelcolor=colors[0])
        axs1.tick_params(axis='x', rotation=45)

        axes = [axs1]

        for i, col in enumerate(sample.columns[1:]):
            ax = axs1.twinx()
            ax.plot(sample.index, sample[col], color=colors[i+1])
            ax.set_ylabel(col, color=colors[i+1])
            ax.tick_params(axis='y', labelcolor=colors[i+1])
            axes.append(ax)
        
        for ax in axes[1:]:
            ax.spines['right'].set_position(("outward", 60 * (axes.index(ax) - 1)))
        
        plt.title('Time series of Operational Features')
        
        fig.tight_layout()

        if not save:
            plt.show()
        else:
            plt.savefig(f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/{name}-{code}-AllOperation-plt.png')
            plt.close()
    

    def make_slider(self, name, code, window_size, feature:str, save_path):

        sample = self.df.loc[(self.df['name'] == name) & (self.df['code'] == code)]

        sample['datetime'] = pd.to_datetime(sample['datetime'])
        sample = sample.sort_values('datetime').reset_index(drop=True)

        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        title = ax.set_title("")

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            start_time = sample.iloc[frame]['datetime']
            end_time = start_time + pd.Timedelta(days=30)
            plot_data = sample[(sample['datetime'] >= start_time) & (sample['datetime'] <= end_time)]

            if plot_data.empty:
                return line,

            line.set_data(plot_data['datetime'], plot_data[feature])
            ax.set_xlim(plot_data['datetime'].min(), plot_data['datetime'].max())
            ax.set_ylim(plot_data[feature].min(), plot_data[feature].max())
            title.set_text(f"{plot_data['datetime'].min().date()} to {plot_data['datetime'].max().date()}")
            return line,

        # Streaming writer (DOES NOT keep all frames in memory)
        writer = FFMpegWriter(fps=5, bitrate=1800)

        ani = FuncAnimation(fig, update, frames=range(len(sample)-window_size), init_func=init, blit=False)

        # Save using the streaming writer
        ani.save(save_path, writer=writer)
        plt.close()
    

    def make_slider2(self, name, code, window_size, feature:str, save_path=None):
        sample = self.df.loc[(self.df['name'] == name) & (self.df['code'] == code)]

        sample['datetime'] = pd.to_datetime(sample['datetime'])
        sample = sample.sort_values('datetime').reset_index(drop=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=sample['datetime'], y=sample[feature], mode='lines', name=feature))

        steps = []

        for i in range(len(sample.index) - window_size):
            start_time = sample['datetime'].iloc[i]
            end_time = start_time + pd.Timedelta(days=window_size)
            plot_data = sample[(sample['datetime'] >= start_time) & (sample['datetime'] <= end_time)]

            step = dict(
                method="update",
                args=[{"x": [plot_data['datetime']], "y": [plot_data[feature]]},
              {"title": f"{start_time.date()} to {end_time.date()}"}],
              label=f"{start_time.date()}"
            )
            steps.append(step)
        
        slider = [dict(
            active=0,
            currentvalue={"prefix": "Start Date: "},
            steps=steps
        )]

        fig.update_layout(
            slider=slider,
            title="Interactive Plot with Slider"
        )

        fig.write_html(save_path)
    
    def simple_plot(self, name, code, x_name:str, y_name:str, save_path=None):

        sample = self.df.loc[(self.df['name'] == name) & (self.df['code'] == code)]

        if x_name == "datetime":
            sample = sample.sort_values(by='datetime')
            sample = sample.set_index('datetime')
            plt.figure(figsize=(48, 24))

            for status, group in sample.groupby('status'):
                plt.scatter(
                    group.index, group[y_name], c=group['color'],
                    label=f'{status} - {len(sample.loc[sample['status']==status].index)}', s=40
                )
            plt.plot(sample.index, sample[y_name], label=f'{y_name} over Time', color='black', linewidth=3)
            plt.xlabel('Time', fontsize=40)
            plt.ylabel(y_name, fontsize=40)
            plt.legend(fontsize=30)
            title = f'{y_name} Over Time - {name}/{code}'
            reshaped_title = reshape(title)
            display_title = get_display(reshaped_title)
            plt.title(display_title, fontsize=60)
            plt.xticks(rotation=90, fontsize=30)
            plt.yticks(fontsize=30)
            plt.grid(True)

            if save_path is None:
                plt.show()
            else:
                plt.savefig(save_path)
                plt.close()

        else:
            
            plt.figure(figsize=(24, 12))
            for status, group in sample.groupby('status'):
                plt.scatter(group[x_name], group[y_name], c=group['color'],
                            label=f'{status} - {len(sample.loc[sample['status']==status].index)}',
                            s=40)
            plt.title(f"{x_name.capitalize()} vs {y_name.capitalize()}")
            plt.xlabel(x_name, fontsize=30)
            plt.ylabel(y_name, fontsize=30)
            plt.text(0.95, 0.05, f'# of points: {len(sample.index)}',
                    fontsize=12, color='black', ha='right', va='bottom', transform=plt.gca().transAxes)
            
            plt.text(0.05, 0.05, f'name: {sample['name'].unique()} - code: {sample['code'].unique()}',
                    fontsize=12, color='black', ha='left', va='top', transform=plt.gca().transAxes)
            
            plt.legend()

            if save_path is None:
                plt.show()
            else:
                plt.savefig(save_path)
                plt.close()