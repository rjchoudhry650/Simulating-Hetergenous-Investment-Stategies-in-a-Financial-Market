#!/usr/local/bin/python

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import string
import math
from Stock import Stock
from Trader import *
from Event import Event

# Runtime expectations, can make new stocks much faster than new traders

# create stock market, an array of stocks
stock_market = []
print("Please input the number of stocks you would like the simulation to run (Max 500):")
num_of_stocks = int(input())
num_of_stocks = min(abs(num_of_stocks), 500)
# set up simulation
print("Please input the number of traders you would like the simulation to run (Max 25): ")
num_of_traders = int(input())
num_of_traders = min(abs(num_of_stocks), 25)
print("Please input the number of days you would like the simulation to run (Max 365): ")
num_of_days = int(input())
num_of_days = min(abs(num_of_days), 365)
chance_of_small_random_event = random.uniform(.01, .09)
chance_of_large_random_event = random.uniform(0.001, .009)
chance_of_none = 1 - chance_of_small_random_event - chance_of_large_random_event
event_probabilities = [chance_of_small_random_event, chance_of_large_random_event, chance_of_none]

directory = os.path.dirname(os.path.realpath(__file__))

stocks_list = pd.read_csv(directory + "/Stock Data/usable_stock_list.csv", index_col=0)

# set up market distribution - based off S&P 500
sectors = []

percent_Miscellaneous = .017
percent_Health_Care = .138
percent_Transportation = .105
percent_Finance = .126
percent_Technology = .242
percent_Capital_Goods = .018
percent_Consumer_Durables = 0.072
percent_Basic_Industries = .09
percent_Consumer_Services = .018
percent_Public_Utilities = 0.035
percent_Energy = .039
percent_Consumer_Non_Durables = 0.098

num_Miscellaneous = math.ceil(percent_Miscellaneous * num_of_stocks)
sectors.append(("Miscellaneous", num_Miscellaneous))
num_Health_Care = math.ceil(percent_Health_Care * num_of_stocks)
sectors.append(("Health Care", num_Health_Care))
num_Transportation = math.ceil(percent_Transportation * num_of_stocks)
sectors.append(("Transportation", num_Transportation))
num_Finance = math.ceil(percent_Finance * num_of_stocks)
sectors.append(("Finance", num_Finance))
num_Technology = math.ceil(percent_Technology * num_of_stocks)
sectors.append(("Technology", num_Technology))
num_Capital_Goods = math.ceil(percent_Capital_Goods * num_of_stocks)
sectors.append(("Capital Goods", num_Capital_Goods))
num_Consumer_Durables = math.ceil(percent_Consumer_Durables * num_of_stocks)
sectors.append(("Consumer Durables", num_Consumer_Durables))
num_Basic_Industries = math.ceil(percent_Basic_Industries * num_of_stocks)
sectors.append(("Basic Industries", num_Basic_Industries))
num_Consumer_Services = math.ceil(percent_Consumer_Services * num_of_stocks)
sectors.append(("Consumer Services", num_Consumer_Services))
num_Public_Utilities = math.ceil(percent_Public_Utilities * num_of_stocks)
sectors.append(("Public Utilities", num_Public_Utilities))
num_Energy = math.ceil(percent_Energy * num_of_stocks)
sectors.append(("Energy", num_Energy))
num_Consumer_Non_Durables = math.ceil(percent_Consumer_Non_Durables * num_of_stocks)
sectors.append(("Consumer Non-Durables", num_Consumer_Non_Durables))

sectors_data = []
for sector in sectors:
    file_location = directory + "/Stock Data/Sectors/" + sector[0] + ".csv"
    stocks_list = pd.read_csv(file_location, index_col=0)
    price_history = []
    for day in range(num_of_days):
        price_history.append(0)
    sectors_data.append([sector[0], sector[1], stocks_list, price_history])

active_sectors = ["Market"]
new_sectors_data = []
for sector in sectors_data:

    num_stocks = sector[1]
    count = 0
    for i in range(num_stocks):
        if len(stock_market) == num_of_stocks:
            break
        count += 1
        length = 4
        name = ''.join(random.choices(string.ascii_letters, k=length)).upper()
        stock_new = Stock(name)
        stock_new.set_new_stock(sector[2], num_of_days)
        stock_market.append(stock_new)
    sector[1] = count

    if count > 0:
        new_sectors_data.append(sector)
        active_sectors.append([sector[0]])

sectors_data = new_sectors_data

# create traders
# implement new trader types
traders = []

trader_types = [
    Trader,
    Day_Trader,
    Pessimistic,
    Trend_Investor,
    Bayesian
]

trader_strategies = [
    ["Rational Trader", 0, 0, []],
    ["Day Trader", 0, 0, []],
    ["Pessimistic Trader", 0, 0, []],
    ["Trend Investor", 0, 0, []],
    ["Bayesian Trader", 0, 0, []]
]


num_traders_per_type = num_of_traders / len(trader_types)
initial_budget = 10000

for trader_type in trader_types:

    for i in range(math.floor(num_of_traders / len(trader_types))):
        trader_new = trader_type(initial_budget)
        trader_new.set_diversification_preference(active_sectors)
        trader_new.update_cash_flow_history(0)
        for stock in stock_market:
            trader_new.set_trader(stock)
        traders.append(trader_new)

# set up simulation loop
event_log = []
active_events = []
event_types = ["small", "large", "none"]

num_buys = 0
num_sells = 0

for day in range(num_of_days):
    print(" ".join(["Day", str(day)]))

    # append to trader_strategies mean performance list
    for strategy in trader_strategies:
        strategy[3].append(0)

    # get event type
    event_type = np.random.choice(event_types, 1, p=event_probabilities)[0]

    if event_type != "none":
        event = Event(day, num_of_days, event_type)
        event.decide_sector(active_sectors)
        event_direction = event.get_direction()
        event_multiplier = event.get_multiplier()
        event_duration = event.get_event_duration()
        print_out = (event_type, event_direction, "shock to", str(event.get_sector()) + ",", "duration of",
                     str(event_duration), "days.")
        print(' '.join(print_out))

        event_log.append([event, day])
        active_events.append(event)

    if len(active_events) == 0:
        event_multiplier = 1

    else:
        for event in active_events:
            event.update_event_duration()
            if event.get_event_duration() <= 0:
                active_events.remove(event)

    for stock in stock_market:

        if stock.get_IPO_Date() > day:
            continue

        # loop through active events
        event_multiplier = 1
        for event in active_events:
            if event.get_sector() == stock.get_sector():
                event_multiplier = event_multiplier * event.get_multiplier()
            if event.get_sector() == "Market":
                event_multiplier = event_multiplier * event.get_multiplier()
        # calculate event multiplier, get gradual increase and descent
        if event_multiplier != 1:
            event_multiplier = event_multiplier / max((event.get_max_point() - day) ** 2, 1)
            event_multiplier = abs((1 + event_multiplier))

        # get current share price
        current_price = stock.get_new_share_price(day, event_multiplier)
        # print("Next Stock:")
        # print(stock.get_name() + " current price on day " + str(day) + ": " + str(current_price))
        sector_performance = 0
        # update sector averages
        for sector in sectors_data:
            if stock.get_sector() == sector[0]:
                sector[3][day] = round(sector[3][day] + (current_price / sector[1]), ndigits=2)
                sector_performance = sector[3]
                continue

        # loop through traders
        for trader in traders:
            # update distribution preferences:
            if day > 2:
                trader.update_diversification_preferences(sectors_data, day)

            decision = "none"
            quantity = 0

            # decide if trader will trade today
            if trader.decide_to_trade(day, num_of_days) and 0 < current_price:

                # evaluate max willingness to pay for stock
                max_WTP = trader.get_max_willingness_to_pay(stock, day, sector_performance)

                # if share price < max willingness to pay, buy stock
                if current_price <= max_WTP:
                    # then get quantity demanded, buy shares
                    quantity = trader.get_quantity_demanded(stock, max_WTP)
                    if quantity > 0 and trader.get_budget() >= current_price:
                        decision = "buy"
                        num_buys += 1

                # if share price > max willingness to pay, and they own stock, sell shares
                elif current_price > max_WTP:
                    if trader.get_num_shares(stock) > 0:
                        # then get quantity supplied, sell shares
                        quantity = trader.get_quantity_supplied(stock, max_WTP)
                        if quantity > 0:
                            decision = "sell"
                            num_sells += 1

            trader.update_after_trade(decision, stock, quantity, day)

print(" ".join(["Number of buys =", str(num_buys)]))
print(" ".join(["Number of sells =", str(num_sells)]))
print(" ".join(["Number of events =", str(len(event_log))]))

# get trader performance averages by trader type
# get number of sales, number of buys by trader type

for strategy in trader_strategies:

    for day in range(num_of_days):
        for trader in traders:
            if trader.get_strategy() == strategy[0]:
                strategy[3][day] = strategy[3][day] + trader.get_total_assets_value_history()[day]

        strategy[3][day] = round(strategy[3][day] / num_traders_per_type, ndigits=2)

# DISPLAY RESULTS:
# Average asset value history for trader type
# Trader returns
# Sector price charts

fig, axs = plt.subplots(2, figsize=(10, 9), num='Simulation Results')
fig.tight_layout(pad=12, h_pad=5, w_pad=3)

for strategy in trader_strategies:
    axs[0].plot(strategy[3], label=str(strategy[0]))

axs[0].set_title('Trader Type Performance Averages')
axs[0].legend(loc='upper right',  bbox_to_anchor=(1.25, 1), borderaxespad=1)
# axs[0].legend(loc=' left', borderaxespad=0)

# axs[0].axhline(y=0, color='r', linestyle='-')
axs[0].set_xlabel("Day")
axs[0].set_ylabel("Value")


for sector in sectors_data:
    axs[1].plot(sector[3], label=sector[0])

axs[1].set_title('Sector Price History')
axs[1].legend(loc='upper right',  bbox_to_anchor=(1.25, 1), borderaxespad=1)
# axs[1].legend(loc='lower left', borderaxespad=0)

axs[1].axhline(y=0, color='r', linestyle='-')
axs[1].set_xlabel("Day")
axs[1].set_ylabel("Price")

plt.show()
