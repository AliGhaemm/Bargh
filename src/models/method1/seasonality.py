import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger
logger = CustomLogger(__name__, log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/seasonality.log').get_logger()

import pandas as pd
import numpy as np
import yaml

class Seasonality:

    def __init__(self):
        logger.info(msg=f"Class instantiated")

    def add_seasonality_effect(self, df: pd.DataFrame):
        try:
            df_c = df.copy()
            ids = df_c['id'].unique().tolist()
            config = self.load_config()

            df_final = pd.DataFrame()

            for id in ids:
                if not id in config['codes']:
                    logger.warning(msg=f"Id {id} is not in the config list ids. check it out!")
                    continue
                df_sampled = df_c[df_c['id'] == id]
                df_sampled = self.add_yearly_effect(df_sampled, config[str(id)]["yearly"]) if config[str(id)]["yearly"] != "None" else df_sampled
                df_sampled = self.add_monthly_effect(df_sampled, config[str(id)]["monthly"]) if config[str(id)]["monthly"] != "None" else df_sampled
                df_sampled = self.add_weekly_effect(df_sampled, config[str(id)]["weekly"]) if config[str(id)]["weekly"] != "None" else df_sampled
                df_sampled = self.add_daily_effect(df_sampled, config[str(id)]["daily"]) if config[str(id)]["daily"] != "None" else df_sampled

                df_final = pd.concat([df_final, df_sampled], ignore_index=True)

            return df_final
        except Exception as e:
            logger.error(msg=f"Seasonality effect did not added. Exception below occured:\n{e}")
            return df
 
    
    def add_yearly_effect(self, df: pd.DataFrame, seasons: list[str]):
        try:
            df_c = df.copy()
            id = df_c['id'].iloc[0]
            df_c['date'] = pd.to_datetime(df_c['date'])
            df_c['month'] = df_c['date'].dt.month

            base_conditions = {
                "winter": df_c['month'].isin([1, 2, 3]),
                "spring": df_c['month'].isin([4, 5, 6]),
                "summer": df_c['month'].isin([7, 8, 9]),
                "fall": df_c['month'].isin([10, 11, 12])
            }

            conditions = [base_conditions[x] for x in seasons]

            choises = list(range(1, len(seasons)+1))

            df_c['year_effect'] = np.select(conditions, choises, default=len(seasons)+1)

            df_c.drop(columns=['month'], inplace=True)

            logger.info(msg=f"Yearly effect successfully added the the dataframe with id {id}.")

            return df_c
        
        except Exception as e:
            logger.error(msg=f"Yearly effect did not added for id {id}. Exception below occured:\n{e}")
            return df

    
    def add_monthly_effect(self):
        pass

    def add_weekly_effect(self, df: pd.DataFrame, days: list[str]):
        try:
            df_c = df.copy()
            id = df_c['id'].iloc[0]
            df_c['date'] = pd.to_datetime(df_c['date'])
            df_c['day'] = df_c['date'].dt.day_name()
            df_c['day'] = df_c['day'].str.lower()

            base_conditions = {
                "saturday": df_c['day'] == "saturday",
                "sunday": df_c['day'] == "sunday",
                "monday": df_c['day'] == "monday",
                "tuesday": df_c['day'] == "tuesday",
                "wednesday": df_c['day'] == "wednesday",
                "thursday": df_c['day'] == "thursday",
                "friday": df_c['day'] == "friday"
            }

            conditions = [base_conditions[x] for x in days]

            choises = list(range(1, len(days)+1))

            df_c['week_effect'] = np.select(conditions, choises, default=len(days)+1)

            df_c.drop(columns=['day'], inplace=True)

            logger.info(msg=f"weekly effect successfully added the the dataframe with id {id}.")

            return df_c

        except Exception as e:
            logger.error(msg=f"Weekly effect did not added for id {id}. Exception below occured:\n{e}")
            return df

    def add_daily_effect(self, df: pd.DataFrame, hours: list[str]):
        try:
            df_c = df.copy()
            id = df_c['id'].iloc[0]
            starts = [int(x.split('-')[0]) for x in hours]
            ends = [int(x.split('-')[1]) for x in hours]

            conditions = [
                df_c['hour'].isin(list(range(starts[i], ends[i]))) for i in range(len(starts)-1)
            ]
            list1 = list(range(starts[-1], ends[-1]))
            list2 = list(range(starts[0]))
            list3 = [*list1, *list2]
            conditions.append(
                df_c['hour'].isin(list3)
            )

            choises = list(range(1, len(hours)+1))

            df_c['hour_effect'] = np.select(conditions, choises, default=len(hours)+1)

            logger.info(msg=f"Daily effect successfully added the the dataframe with id {id}.")

            return df_c
        
        except Exception as e:
            logger.error(msg=f"Daily effect did not added for id {id}. Exception below occured:\n{e}")
            return df

    def load_config(self):
        try:
            path = "/home/hajali/Desktop/Bargh_Ml_project/configs/seasonality.yaml"
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            pass