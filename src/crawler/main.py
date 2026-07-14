import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from crawl_history import HistoryCrawler
from crawl_tommorow import ForecastCrawler
from src.data.cleaning.data_cleaning import RawData

if __name__ == "__main__":
    print(RawData.PLANT.value)
    # hc = HistoryCrawler(file=RawData.PLANT.value, is_csv=False)
    hc = HistoryCrawler(file=RawData.PLANT.value)
    hc.crawl(
        start_date='2021-03-21',
        end_date="2025-03-20",
        is_csv=True
    )
    # fc = ForecastCrawler(file=RawData.PLANT.value)
    # fc.crawl()