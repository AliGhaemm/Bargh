import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger

logger = CustomLogger(__name__, '/home/hajali/Desktop/Bargh_Ml_project/logs/testlog.log').get_logger()

logger.info(msg='Testing number 1')
logger.error(msg='Testing number 2')

from src.data.dbconnection import Database

db = Database()
db.connect()
result = db.execute(query='select * from final limit 10')
print(result)

c = db.get_cursor()

c.execute(query='select * from final limit 10')