import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from src.data.dbconnection import Database
from logs.logger import CustomLogger

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime
from convertdate import persian
import yaml

logger = CustomLogger('Historical', log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/HistoricalCrawler.log').get_logger()   

db = Database()

feature_dict = yaml.load(open('/home/hajali/Desktop/Bargh_Ml_project/configs/tables_columns.yaml'), Loader=yaml.SafeLoader)

class HistoryCrawler:
    def __init__(self, file: str):
        self.file = file
    
    def get_innermost_dict(self, nested_dict: dict):
        if not isinstance(nested_dict, dict):
            return None

        if not nested_dict:
            return None

        first_value = next(iter(nested_dict.values()))

        if isinstance(first_value, dict):
            return self.get_innermost_dict(first_value)
        else:
            return nested_dict
    
    def get_plants(self):
        data = pd.read_csv('/home/hajali/Desktop/Bargh_Ml_project/data/raw/PlantsTemperature_View.csv')
        plants = list(map(str, list(data['PowerPlantCode'].drop_duplicates())))
        return plants
    
    def crawl(self, start_date: str, end_date: str, is_csv=True):

        try:
            plants = pd.read_csv(filepath_or_buffer=self.file) if is_csv else pd.read_excel(self.file)
            my_plants = self.get_plants()
            print(my_plants)
            print(type(my_plants[0]))
            print(type(plants['DispPlantCode'].iloc[0]))
            plants = plants[plants['DispPlantCode'].isin(my_plants)]
            print(plants.head())

            logger.info(msg=f'Plants data successfully read from {self.file}')
            logger.info(msg=f"Plants to crawl:\n{plants['PlantName'].drop_duplicates()}")

            do_continue = input("Continue with these plants?\n")
            if do_continue == "no":
                return None

            with open('/home/hajali/Desktop/Bargh_Ml_project/configs/crawling.yaml', 'r') as file:
                data = yaml.safe_load(file)
                url = data['url-historical']
                hourly_features = data['hourly']

            logger.info(msg=f'Successfully read data from crawling config file')

            cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
            retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
            openmeteo = openmeteo_requests.Client(session = retry_session)

            data = pd.DataFrame()

            for utm, unitid in zip(plants['UTM'], plants['DispPlantCode']):
                lat, longit = utm.split(',')
                params = {
                    "latitude": float(lat),
                    "longitude": float(longit),
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": hourly_features
                }
                responses = openmeteo.weather_api(url, params=params)

                response = responses[0]

                print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
                print(f"Elevation {response.Elevation()} m asl")
                print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
                print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

                # Process hourly data. The order of variables needs to be the same as requested.
                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
                hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
                hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
                # hourly_precipitation_probability = hourly.Variables(4).ValuesAsNumpy()
                hourly_precipitation = hourly.Variables(5).ValuesAsNumpy()
                hourly_rain = hourly.Variables(6).ValuesAsNumpy()
                hourly_snowfall = hourly.Variables(7).ValuesAsNumpy()
                hourly_surface_pressure = hourly.Variables(8).ValuesAsNumpy()
                hourly_evapotranspiration = hourly.Variables(9).ValuesAsNumpy()
                hourly_wind_speed_10m = hourly.Variables(10).ValuesAsNumpy()
                hourly_wind_direction_10m = hourly.Variables(11).ValuesAsNumpy()

                hourly_data = {"date": pd.date_range(
                    start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                    end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                    freq = pd.Timedelta(seconds = hourly.Interval()),
                    inclusive = "left"
                )}

                hourly_data["temperature_2m"] = hourly_temperature_2m
                hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
                hourly_data["dew_point_2m"] = hourly_dew_point_2m
                hourly_data["apparent_temperature"] = hourly_apparent_temperature
                # hourly_data["precipitation_probability"] = hourly_precipitation_probability
                hourly_data["precipitation"] = hourly_precipitation
                hourly_data["rain"] = hourly_rain
                hourly_data["snowfall"] = hourly_snowfall
                hourly_data["surface_pressure"] = hourly_surface_pressure
                hourly_data["evapotranspiration"] = hourly_evapotranspiration
                hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
                hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
                hourly_data['unitid'] = [unitid for _ in range(len(hourly_temperature_2m))]

                hourly_dataframe = pd.DataFrame(data=hourly_data)

                data = pd.concat([data, hourly_dataframe], ignore_index=True)
                logger.info(msg=f'Data with latitude: {lat} and longitude: {longit} added to the dataframe')
        
            data['date'] = pd.to_datetime(data['date'])
            logger.info(msg=f"Date column converted to datetime type")

            data['date_only'] = data['date'].dt.date
            data['time'] = data['date'].dt.time
            data['time'] = data['time'].apply(lambda x: int(str(x).split(':')[0]))
            logger.info(msg=f"Create the date and time columns from date column")

            data.loc[data['time'] == 0, 'time'] = 24
            data.loc[data['time'] == 24, 'date_only'] = data['date_only'] - pd.Timedelta(days=1)
            logger.info(msg=f"Time column corrected")

            data.drop(columns=['date'], axis=1, inplace=True)
            logger.info(msg=f"Column date deleted")

            # data['date_only'] = data['date_only'].apply(self.convert_to_solar)
            logger.info(msg=f"Date column converted from Ad to Persian")

            data.rename(columns={'date_only': 'date'}, inplace=True)

            columns_to_move = ['unitid', 'date', 'time']

            new_columns = columns_to_move + [col for col in data.columns if col not in columns_to_move]

            data = data[new_columns]

            l = len(data.index)

            data.dropna(inplace=True)

            print(f"{l - len(data.index)} number of data removed because of being null.")
            
            logger.info(msg=f"Reorder the columns as the id, date and time comes to first.")

            file_path = '/home/hajali/Desktop/Bargh_Ml_project/data/interim/weather.csv'

            data.to_csv(file_path, index=False)

            logger.info(msg=f"Number of columns of dataframe is: {len(data.columns)}")

            db.connect()
            
            db.create_table(
                table_name='weather',
                columns=self.get_innermost_dict(feature_dict['weather'])
            )
            db.commit()

            db.lazy_copy_expert(
                table_name='weather',
                file=file_path,
                mode='r',
                into_db=True
            )
            db.commit()
            db.close()

        except Exception as e:
            logger.error(msg=f"Couldnt complete the crawling due to below Exception:\n{e}\n")
    

    def convert_to_solar(self, AD_date):
        ad_date = datetime.strptime(str(AD_date), '%Y-%m-%d')
        solar_date = persian.from_gregorian(ad_date.year, ad_date.month, ad_date.day)
        return f'{solar_date[0]}/{solar_date[1]:02d}/{solar_date[2]:02d}'