import pandas as pd
import numpy as np
import os

# split based on sector
# need list of sectors
# import list


def get_sectors_lists():

    directory = os.path.dirname(os.path.realpath(__file__))
    stocks_list = pd.read_csv(directory + "/Stock Data/usable_stock_list.csv", index_col=0)

    sectors = [("Miscellaneous", []), ("Health Care", []), ("Transportation", []), ("Finance", []), ("Technology", []),
               ("Capital Goods", []),
               ("Consumer Durables", []),
               ("Basic Industries", []),
               ("Consumer Services", []),
               ("Public Utilities", []),
               ("Energy", []),
               ("Consumer Non-Durables", [])]

    count = 1
    for row in range(len(stocks_list)):
        print(count)
        count += 1
        industry = stocks_list.loc[row, 'Sector']

        if pd.isnull(industry):
            stocks_list.loc[row, 'Sector'] = "Miscellaneous"
            industry = "Miscellaneous"

        for sector in sectors:

            if sector[0] == industry:
                symbol = stocks_list.loc[row, 'Symbol']
                sector[1].append([symbol, industry])


    for sector in sectors:
        sector_list = pd.DataFrame(sector[1], columns=['Symbol', 'Sector'])
        file_path = directory + "/Stock Data/Sectors/"
        file_name = sector[0] + ".csv"
        file_name = file_path + file_name
        sector_list.to_csv(file_name)


