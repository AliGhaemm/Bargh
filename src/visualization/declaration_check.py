import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import convertdate
from provide_DecChck import ProvideData



class DeclareCheck:
    def __init__(self):
        file_path = '/home/hajali/Desktop/Bargh_Ml_project/data/processed/decChck_data.csv'

        if os.path.exists(file_path):
            self.df = pd.read_csv(file_path)
        else:
            provider = ProvideData()
            provider.provide()
            self.df = pd.read_csv(file_path)

    def jalali_to_gregorian(self, date_str):
        try:
            year, month, day = map(int, date_str.split('-'))
            g_year, g_month, g_day = convertdate.persian.to_gregorian(year, month, day)
            return f'{g_year}-{g_month:02d}-{g_day:02d}'
        except Exception as e:
            print(f'Exception occured: {e}')
    
    def plot_powerplant(self, data: pd.DataFrame, unitname: str, unitcode: str, do_save=False):
        data = data.set_index('datetime')
        plt.figure(figsize=(30, 20))
        plt.plot(data.index, data['declare'], label='Declaration', color='red', marker='o')
        plt.plot(data.index, data['calculated'], label='Expected', color='blue', marker='*')
        plt.xlabel('Time', fontsize=25)
        plt.ylabel('Power', fontsize=25)
        plt.title('Declaration vs Expected', fontsize=35)
        plt.legend(fontsize=30)
        plt.xticks(rotation=90, fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(True)
        if not do_save:
            
            plt.show()
        else:
            plt.savefig(f'/home/hajali/Desktop/Bargh_Ml_project/src/visualization/declaration_check/{unitname}-{unitcode}-decChck.png')
            plt.close()

    def check_unit(self, unitname: str, unitcode: str, do_save=False):
        
        data = self.df[(self.df['name'] == unitname) & (self.df['code'] == unitcode)]

        data['date'] = data['date'].apply(self.jalali_to_gregorian)
        data['date'] = pd.to_datetime(data['date'])

        data['datetime'] = data['date'] + pd.to_timedelta(data['hour'], unit='H')

        self.plot_powerplant(
            data=data,
            unitname=unitname,
            unitcode=unitcode,
            do_save=do_save
        )
    
    def check_all(self, do_save=False):
        pass