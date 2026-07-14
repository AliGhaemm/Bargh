import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from src.data.dbconnection import Database
from logs.logger import CustomLogger

import yaml

logger = CustomLogger(name='__main__', log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/DecChck_data.log').get_logger()

class ProvideData:
    def __init__(self):
        self.db = Database()
        self.db.connect()
        self.db.__exit__()

    def provide(self):
        try:
            table_configs = self.load_table_configs(filename='/home/hajali/Desktop/Bargh_Ml_project/configs/tables_columns.yaml')

            sql_template = self.load_sql_query(filename='/home/hajali/Desktop/Bargh_Ml_project/src/data/queries/decChck.sql')

            logger.info(msg=f"Successfully loaded table configs and sql template.")

            target_table = list(table_configs['declaration_check'].keys())[0]

            sql_query = sql_template.format(
                target_table = target_table,
                id = list(table_configs['declaration_check'][target_table].keys())[0],
                name = list(table_configs['declaration_check'][target_table].keys())[1],
                code = list(table_configs['declaration_check'][target_table].keys())[2],
                date = list(table_configs['declaration_check'][target_table].keys())[3],
                hour = list(table_configs['declaration_check'][target_table].keys())[4],
                a = list(table_configs['declaration_check'][target_table].keys())[5],
                b = list(table_configs['declaration_check'][target_table].keys())[6],
                temperature = list(table_configs['declaration_check'][target_table].keys())[7],
                declare = list(table_configs['declaration_check'][target_table].keys())[8],
                aggregation_table = 'operational',
                factor_table = list(table_configs['factors'].keys())[0],
                status_type = list(table_configs['status']['status'].keys())[3]
            )

            self.db.__enter__()
            self.db.execute(
                query=sql_query,
                do_return=False
            )
            self.db.commit()

            logger.info(msg=f"Successfully created the table for declaration check.")
            
            self.db.lazy_copy_expert(
                table_name=target_table,
                file='/home/hajali/Desktop/Bargh_Ml_project/data/processed/decChck_data.csv',
                mode='w',
                into_local=True
            )
            self.db.commit()

            self.db.__exit__()
        
        except Exception as e:
            logger.error(msg=f'Couldnt provide the table for declaration checking. the Exception below occured:\n{e}\n')

    def load_sql_query(self, filename):
        with open(filename, 'r') as f:
            return f.read()
    
    def load_table_configs(self, filename):
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
