import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
import pandas as pd
import datetime as dt
import numpy as np
import LSTM
import os
import matplotlib.pyplot as plt
from matplotlib import style
import Usable_Names_List as usable


def get_web_data():
    print("Running web scraper")
    print(f"This will take significant time, potentially 30-45 hours."
          f" Data from 1/1/2017 to 11/20/2021 is collected for\n"
          "each stock, then split into test and train data, and "
          "ran through the LSTM to create extrapolative dataset for \n"
          "365 days in the future. This will repeat for over 4000 stocks. "
          "The resulting data is already availble in the\n"
          "Extrapolation folder.")
    start = dt.datetime(2017, 1, 1)
    end = dt.datetime(2021, 11, 20)

    # RUNTIME IS VERY LONG, MAY TAKE 30-35 HOURS

    directory = os.path.dirname(os.path.realpath(__file__))
    file = open(directory + "/Stock Data/NASDAQ_Stock_Names.csv")
    industry_data = pd.read_csv(file)

    extrapolation = LSTM
    batch = len(industry_data)
    last_run = 1
    count = last_run

    for row in range(last_run, batch):
            symbol = industry_data.loc[row, 'Symbol']
            sector = industry_data.loc[row, 'Sector']
            print(str(round(count/batch, ndigits=2)) + "%: " + symbol)
            print("Count: " + str(count))
            count += 1
            try:
                prices = web.DataReader(symbol, 'yahoo', start, end)['Close']
                print(len(prices))
                if len(prices) > 900:
                    # convert list into pandas dataframe, pass as dataframe
                    df = pd.DataFrame({'Date': prices.index, 'Close': prices.values})
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df = df['Close']

                    extrapolation.forecast_stock_price(symbol, df)

                    print("Usable data = " + str(len(data_list)))
            except KeyError:
                pass
            except RemoteDataError:
                pass

    # need to make usable list of available stocks
    print("Getting usable stocks data list")
    Usable = usable
    Usable.create_usable_stocks_list(industry_data)






