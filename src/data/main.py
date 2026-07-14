import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from cleaning.data_cleaning import csvfile_manipulation, RawData
from aggregation import Aggregator

if __name__ == "__main__":
    # manipulator = csvfile_manipulation()
    # for raw in RawData:
    #     if raw == RawData.STATUS:
    #         manipulator.process(file=raw)
        

        
    
    aggregator = Aggregator(name='hajali')

    aggregator.operational_aggregation()