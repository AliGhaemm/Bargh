import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from dbconnection import Database
from logs.logger import CustomLogger
import yaml
from psycopg2 import Error as Error

logger = CustomLogger(__name__, log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/aggregation.log').get_logger()

db = Database()

class Aggregator:
    def __init__(self, name: str):
        self.user = name
        self.db = Database()
        db.connect()
        db.__exit__()
        self.logger = CustomLogger(__name__, log_gile=f'/home/hajali/Desktop/Bargh_Ml_project/logs/aggregation({name}).log').get_logger()
        self.query_path = '/home/hajali/Desktop/Bargh_Ml_project/src/data/queries/'
    
    def operational_aggregation(self):
        try:
            target_table = 'operational'

            table_config = self.load_tables_configs('/home/hajali/Desktop/Bargh_Ml_project/configs/tables_columns.yaml')

            sql_template = self.load_sql_query('/home/hajali/Desktop/Bargh_Ml_project/src/data/queries/operational.sql')

            logger.info(msg=f'Successfully loaded table configs and sql template.')

            sql_query = sql_template.format(
                target_table = target_table,
                weather_table = list(table_config['weather'].keys())[0],
                temp_table = list(table_config['plant_temp'].keys())[0],
                bar_table = list(table_config['bar'].keys())[0],
                energy_table = list(table_config['energy'].keys())[0],
                seller_table = list(table_config['selleroffer'].keys())[0],
                status_table = list(table_config['status'].keys())[0],
                id = list(table_config['energy']['energy'].keys())[0],
                name = list(table_config['energy']['energy'].keys())[2],
                code = list(table_config['energy']['energy'].keys())[1],
                value = list(table_config['bar']['bar'].keys())[2],
                date = list(table_config['energy']['energy'].keys())[3],
                hour = list(table_config['energy']['energy'].keys())[4],
                generation = list(table_config['energy']['energy'].keys())[5],
                declare = list(table_config['selleroffer']['selleroffer'].keys())[5],
                status_type = list(table_config['status']['status'].keys())[4],
                load_table = list(table_config['load'].keys())[0],
                forecast = list(table_config['load']['load'].keys())[2],
                commitment_table = list(table_config['commitment'].keys())[0],
                require = list(table_config['commitment']['commitment'].keys())[5]
            )

            db.__enter__()
            db.execute(
                query=sql_query,
                do_return=False
            )
            db.commit()

            logger.info(msg=f'Successfully applied the query:\n{sql_query}\n on database.')

            db.lazy_copy_expert(
                table_name=target_table,
                file='/home/hajali/Desktop/Bargh_Ml_project/data/processed/operation.csv',
                mode='w',
                into_local=True
            )

            db.commit()

            db.__exit__()

        except Error as e:
            logger.error(f'Couldnt apply the query:\n{sql_query}\n Exception:\n{e}\n occured.')
        except Exception as exc:
            logger.error(f'Couldnt apply the query:\n{sql_query}\n Exception:\n{exc}\n occured.')


    def load_sql_query(self, filename):
        with open(filename, 'r') as f:
            return f.read()
    
    def load_tables_configs(self, filename):
        with open(filename, 'r') as f:
            return yaml.safe_load(f)