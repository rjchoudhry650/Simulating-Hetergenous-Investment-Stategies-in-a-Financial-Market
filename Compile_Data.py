import pandas as pd
import os
import web_scraper
import Sectors

# Necessary Packages:
# pandas, numpy, pandas-datareader, matplotlib, keras, sklearn

def main():
    print("running")
    get_NASDAQ_names_list()
    web_data = web_scraper
    web_data.get_web_data()
    sector_maker = Sectors
    sector_maker.get_sectors_lists()


def get_NASDAQ_names_list():
    print("Making NASDAQ list")
    directory = os.path.dirname(os.path.realpath(__file__))
    industry_data = pd.read_csv(directory + "/Stock Data/Original Data/NASDAQ_Industry_Data.csv")
    industry_data = industry_data[['Symbol', 'Sector']]
    industry_data.to_csv(directory + "/Stock Data/NASDAQ_Stock_Names2.csv")


main()
