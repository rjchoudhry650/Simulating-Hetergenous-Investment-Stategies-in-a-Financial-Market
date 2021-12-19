import random
import math
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web


class Stock:

    def __init__(self, name):
        # basic attributes
        self.name = name
        self.sector = "None"
        self.directory = os.path.dirname(os.path.realpath(__file__))

        # price attributes
        self.IPO = 0
        self.IPO_date = 0
        self.data = []
        self.share_price = 0
        self.share_price_history = []
        self.standard_deviation = 0
        self.bankrupt = False

# sim initialize functions
    def set_new_stock(self, stocks_list, num_of_days):
        # import list of stocks as npy
        # choose a random one
        random_row = random.randint(0, len(stocks_list) - 1)
        stock_name = stocks_list.loc[random_row, 'Symbol']
        self.sector = stocks_list.loc[random_row, 'Sector']
        # pass to load_data
        self.load_data(stock_name)
        self.set_IPO_date(num_of_days)

    def load_data(self, stock_name):
        file_location = self.directory + "/Stock Data/Extrapolation/" + stock_name + ".npy"
        self.data = np.load(file_location)
        self.standard_deviation = np.std(self.data)
        # print(self.standard_deviation)

    def set_IPO_date(self, num_of_days):
        # IPO random for now
        if len(self.data) < num_of_days:
            # get days of data available
            self.IPO_date = num_of_days - len(self.data) + 1

        self.IPO = self.data[self.IPO_date][0]

# basic functions
    def get_name(self):
        return self.name

    def get_sector(self):
        return self.sector

    def get_IPO(self):
        return self.IPO

    def get_IPO_Date(self):
        return self.IPO_date

    # price functions
    def get_new_share_price(self, day, event_multiplier):
        day = day - self.get_IPO_Date()
        price = float(self.data[day][0])
        # solution, set IPO for middle of simulation
        # add code for if next index is out of range, use last price
        price = (price + self.get_variation()) * abs((1 + event_multiplier))
        price = round(price, ndigits=2)
        if price <= 0:
            self.bankrupt = True

        if self.bankrupt:
            price = 0
        self.share_price = price
        self.update_share_price_history(price)

        return price

    def get_share_price(self):
        return self.share_price

    def get_share_price_history(self):
        return self.share_price_history

    def update_share_price_history(self, price):
        self.share_price_history.append(price)

    def get_variation(self):
        upper_bound = (self.standard_deviation * .8)
        lower_bound = (0 - self.standard_deviation * .8)
        variation = random.uniform(lower_bound, upper_bound)
        return variation


