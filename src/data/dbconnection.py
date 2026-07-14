import yaml
import psycopg2
from logs.logger import CustomLogger

logger = CustomLogger(__name__, log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/database.log').get_logger()

class Database:
    def __init__(self):
        self.connection_parameters = {
            key: value for key, value in yaml.load(open('/home/hajali/Desktop/Bargh_Ml_project/configs/database.yaml'), Loader=yaml.SafeLoader).items()
            }
        self.connection = None
    
    def connect(self):
        if self.connection is None:
            try:
                self.connection = psycopg2.connect(**self.connection_parameters)
                logger.info(f'a user connected to DB.')
            except Exception as e:
                logger.error(f'Couldnt connect to the DB. Exception: \n{e}\n occured.')
    
    def close(self):
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                logger.info(f'Connection to the DB cloesd.')
            except Exception as e:
                logger.error(f'Couldnt disconnect, Exception \n{e}\n occured.')
    
    def execute(self, query:str, params=None, do_return=False):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                logger.info(f'Successfully execute the query: \n{query}\n on the DB.')
                if do_return:
                    return cursor.fetchall()
            
        except Exception as e:
            logger.error(f'Couldnt execute query: \n{query}\n on the DB. Exception \n{e}\n occured.')
    
    def copy_expert(self, query: str, file, mode: str):
        try:
            with self.connection.cursor() as cursor:
                with open(file, mode) as f:
                    cursor.copy_expert(query, f)
                logger.info(f'Successfully copied file {file}')
        except Exception as e:
            logger.error(f'Couldint copy the file with query:\n{query}\n because the Exception\n{e}\n occured.')
    
    def commit(self):
        try:
            if self.connection:
                self.connection.commit()
                logger.info('Commitment successfully applied.')
        except Exception as e:
            logger.error(f'Commitment failed.')
    
    def rollback(self):
        try:
            if self.connection:
                self.connection.rollback()
                logger.info('Rollback successfully applied.')
        except Exception as e:
            logger.error(f'Rollback couldnt apply. Exception\n{e}\n occured.')
    
    def __enter__(self):
        try:
            self.connect()
            logger.info('User reConnected to the DB')
            return self
        except Exception as e:
            logger.error('Can not Enter the DB. Exception\n{e}\n occured.')
    
    def __exit__(self):
        try:
            self.close()
            logger.info('User exited from DB.')
        except Exception as e:
            logger.error('Can not exit the DB. Exception\n{e}\n occured.')
    
    def get_cursor(self):
        return self.connection.cursor()
    
    def create_table(self, table_name: str, columns: dict[str: str]):
        features = [f'{col_name} {col_type}' for col_name, col_type in columns.items()]
        converted = ', '.join(features)
        self.execute(
            query=f'create table if not exists {table_name} ({converted})',
            do_return=False
        )

    def lazy_copy_expert(self, table_name: str, file: str, mode:str, into_db=False, into_local=False):
        if into_db:
            self.copy_expert(
                query=f"copy {table_name} from stdin with delimiter ',' csv header NULL as 'NULL'",
                file=file,
                mode=mode
            )
        if into_local:
            self.copy_expert(
                query=f"copy {table_name} to stdout with delimiter ',' csv header NULL as 'NULL'",
                file=file,
                mode=mode
            )