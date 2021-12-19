import pandas as pd
import numpy as np
from os.path import exists
import os

# get usable stock names list
# use original full list, eliminate if file exists in repository
# append to current list, made in web_scraper

# first remove cross referenced from data


def create_usable_stocks_list(industry_data):

    usable_list = []

    directory = os.path.dirname(os.path.realpath(__file__))
    stock_data_list = os.listdir(directory + "/Stock Data/Extrapolation")

    count = 1

    for stock in stock_data_list:
        print(count)
        last_char = (len(stock) - 4)
        stock = stock[0:last_char]

        for row in range(len(industry_data)):
            symbol = industry_data.loc[row, 'Symbol']
            sector = industry_data.loc[row, 'Sector']

            if stock == symbol:
                usable_list.append([symbol, sector])

        count += 1

    print(str(len(stock_data_list)))
    print(str(len(usable_list)))
    df = pd.DataFrame(usable_list, columns=['Symbol', 'Sector'])
    df.to_csv(directory + "/Stock Data/usable_stock_list.csv")
