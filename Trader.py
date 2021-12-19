import math
import numpy as np
import matplotlib.pyplot as plt
import random
import string

# Investor types:
# RATIONAL, TREND, PESSIMISTIC, DAY TRADER, BAYESIAN TRADER


class Trader:
    def __init__(self, budget):
        # basic attributes
        self.initial_budget = budget
        self.budget = budget
        self.strategy = "Rational Trader"
        # rational trader perceives all information with complete accuracy, looks at every market and every sector,
        # diversifies portfolio, uses all of budget ,is non speculative ,trades infrequently, splits portfolio uniformly
        self.bankrupt = False

        # trader preferences
        self.risk_preference = random.uniform(.01, 1)
        self.trade_frequency = 52
        self.speculation = 0
        self.diversification_preferences = {}

        # shareholder attributes
        self.portfolio = {}
        # total value of held assets at any given time
        self.portfolio_value_history = []
        self.cash_flow_history = []
        # FIX: maybe in future, breakdown value and returns by industry, respond more to industry with higher gains
        # aggregate value of returns at any given time
        self.returns_history = []
        self.total_assets_value_history = []

    # sim initialize functions
    def set_trader(self, stock):
        self.portfolio[stock] = [0, 0]

    def set_diversification_preference(self, active_sectors):
        sectors = active_sectors.copy()
        sectors.remove("Market")

        for sector in sectors:
            sector = sector[0]
            self.diversification_preferences[sector] = 1 / (len(sectors))

    # basic get functions
    def get_budget(self):
        return self.budget

    def get_strategy(self):
        return self.strategy

    def get_bankrupt(self):
        return self.bankrupt

    def get_speculation(self, day):
        return self.speculation

    def get_portfolio(self):
        return self.portfolio

    def get_portfolio_value(self):
        return self.portfolio_value_history[-1]

    def get_cash_flow_history(self):
        return self.cash_flow_history

    def get_portfolio_value_history(self):
        return self.portfolio_value_history

    def get_returns_history(self):
        return self.returns_history

    def get_total_assets_value_history(self):
        return self.total_assets_value_history

    def get_num_shares(self, stock):
        num_shares = self.portfolio[stock][0]
        return num_shares

    def get_last_returns(self):
        return self.returns_history[-1]

    # basic update functions
    def update_diversification_preferences(self, sectors_data, day):
        pass

    def update_after_trade(self, decision, stock, shares, day):
        # portfolio -> shares = [0]
        # portfolio -> buy price = [1]
        price = stock.get_share_price()
        if decision == "buy":
            # update budget
            self.budget = round(self.budget - (price * shares), ndigits=2)
            # update portfolio -> shares
            self.portfolio[stock][0] = self.portfolio[stock][0] + shares

            # update portfolio -> buy price
            if self.portfolio[stock][1] > 0:
                self.portfolio[stock][1] = (self.portfolio[stock][1] + price) / 2

            else:
                self.portfolio[stock][1] = price

        elif decision == "sell":
            # update budget
            self.budget = round(self.budget + (price * shares), ndigits=2)
            # update portfolio
            self.portfolio[stock][0] = self.portfolio[stock][0] - shares

            # update returns history
            self.update_returns_history(stock, price, shares, day)

        # update cash flow history
        self.update_cash_flow_history(day)
        # update portfolio value history
        self.update_portfolio_value_history(day)
        # update asset value history
        self.update_total_assets_value_history(day)
        # update returns history
        self.update_returns_history(stock, price, shares, day)
        # check for bankruptcy
        self.update_bankrupt()

    def update_cash_flow_history(self, day):
        if len(self.cash_flow_history) > day:
            self.cash_flow_history[day] = self.budget

        else:
            self.cash_flow_history.append(self.budget)

    def update_returns_history(self, stock, price, shares, day):
        # returns = sale price * shares - buy price - shares
        returns = (price - self.get_portfolio()[stock][1]) * shares

        if len(self.returns_history) > day:
            self.returns_history[day] = self.returns_history[day] + returns
        else:
            self.returns_history.append(returns)

    def update_portfolio_value_history(self, day):
        # gets total value of portfolio
        total_value = 0
        for stock in self.portfolio:
            share_value = stock.get_share_price() * self.portfolio[stock][0]
            total_value = total_value + share_value

        if len(self.portfolio_value_history) > day:
            self.portfolio_value_history[day] = total_value
        else:
            self.portfolio_value_history.append(total_value)

    def update_total_assets_value_history(self, day):
        # gets total value of portfolio
        total_value = 0

        total_value = self.portfolio_value_history[day] + self.cash_flow_history[day]

        if len(self.total_assets_value_history) > day:
            self.total_assets_value_history[day] = total_value
        else:
            self.total_assets_value_history.append(total_value)

    def update_bankrupt(self):
        if self.budget <= 0 and self.portfolio_value_history[-1] <= 0:
            self.bankrupt = True

        else:
            self.bankrupt = False

    # trading functions
    def evaluate_risk(self, stock):
        # test with random input
        risk = np.std(stock.get_share_price_history())
        risk = (1 - self.risk_preference) * risk
        # if risk = 0, means they have no risk concerns
        risk = max(.01, risk)
        return risk

    def estimate_future_earnings(self, stock):
        performance = stock.get_share_price_history()
        average_performance = sum(performance) / len(performance)
        last_price = stock.get_share_price()
        growth_potential = (last_price - average_performance) / average_performance
        expected_growth = last_price * (growth_potential + 1)

        return expected_growth

    def decide_to_trade(self, day, num_of_days):
        trade_on = math.ceil(num_of_days / self.trade_frequency)
        trade = False
        if day == 0:
            return trade

        if day % trade_on == 0:
            trade = True
        return trade

    def evaluate_sector_performance(self, stock, sector_performance, day):
        stock_performance = stock.get_share_price_history()
        # compare historical averages

        stock_average_performance = sum(stock_performance) / len(stock_performance)
        sector_average_performance = sum(sector_performance) / len(sector_performance)
        if stock_average_performance == 0:
            return 0
        stock_day = min(day, len(stock_performance))
        stock_growth_potential = (stock_performance[
                                      stock_day - 1] - stock_average_performance) / stock_average_performance
        sector_growth_potential = (sector_performance[
                                       day - 1] - sector_average_performance) / sector_average_performance

        if stock_growth_potential == 0:
            return 0

        growth_differential = (stock_growth_potential - sector_growth_potential) / sector_growth_potential
        return growth_differential

    # uses risk and future earnings evaluation to
    def get_max_willingness_to_pay(self, stock, day, sector_performance):
        # expected future value of the stock over
        # value in comparison other stocks in sector - performs higher than average, want to buy more
        sector = self.evaluate_sector_performance(stock, sector_performance, day)
        budget = (self.get_budget() - self.initial_budget) / self.initial_budget
        risk = self.evaluate_risk(stock)
        estimated_earnings = self.estimate_future_earnings(stock)
        # update distribution preferences:
        distribution = self.distribution_preference(stock.get_sector())
        speculation = self.get_speculation(day)

        max_WTP = min(
            ((estimated_earnings * (1 + speculation)) / risk) * distribution * (1 + budget) +
            (1 + sector) * stock.get_share_price()
            , self.budget)
        max_WTP = max(round(max_WTP, ndigits=2), 0)

        return max_WTP

    def distribution_preference(self, sector):
        # increases quantity demanded if current distribution of stocks unfavorable
        sector_shares = 0
        total_shares = 0
        for stock in self.portfolio:
            total_shares += self.portfolio[stock][0]
            if stock.get_sector() == sector:
                sector_shares += self.portfolio[stock][0]

        if total_shares == 0:
            return 1
        ratio = sector_shares / total_shares
        diversification_preference = self.diversification_preferences[sector]
        difference = ratio - diversification_preference
        return 1 - difference

    def get_quantity_demanded(self, stock, max_WTP):
        # incorporate budget so that trader can never over purchase stock
        # also a function of current shares held
        price = stock.get_share_price()
        budget = self.get_budget()
        distribution = self.diversification_preferences[stock.get_sector()]
        # quantity = math.floor(budget - (price * random.randint(0, 10)))
        # quantity = max(quantity, 0)

        # higher risk, lower demand
        # higher expected return, higher demand
        # demand certain amount from each sector (can save for portfolio optimizer, rational doesn't care about sector)
        # rational investor would want to diversify across market, buy equal proportion of each

        quantity = math.ceil((
                                     (max_WTP - price) / price) * (distribution * budget) / price
                             )

        if quantity * price > budget:
            quantity = math.floor(budget / price)

        return quantity

    def get_quantity_supplied(self, stock, max_WTP):
        # max quantity supplied is num shares
        # based on current max willingness to pay
        price = stock.get_share_price()
        max_supply = self.portfolio[stock][0]
        buy_price = self.portfolio[stock][1]

        # supply = math.floor(max_supply - (budget / price))

        if max_WTP <= 0:
            supply = max_supply

        else:
            supply = math.floor((max_supply * min(1, abs(price - max_WTP) / max_WTP)) * \
                                (min(1, abs((buy_price - price) / buy_price))))

        return supply


# TREND INVESTOR CLASS
class Trend_Investor(Trader):

    def __init__(self, budget):
        super().__init__(budget=budget)
        self.strategy = "Trend Investor"
        self.budget = budget
        # chases trends ove a larger time horizon, and trades less frequently, than a day trader
        # trades twice weekly, anchors to 52 week max, believes stock will rebound, want to buy more if farther from max
        # slightly more risk than rational investor
        self.risk_preference = random.uniform(.25, 1)
        self.trade_frequency = 104

    # evaluate future earnings in short time horizon
    def estimate_future_earnings(self, stock, day):
        performance = stock.get_share_price_history().copy()
        look_back = max(day - 30, 0)

        performance = performance[look_back: day]
        average_performance = sum(performance)/max(len(performance), look_back)
        if average_performance == 0:
            return 0

        last_price = stock.get_share_price()
        growth_potential = (last_price - average_performance) / average_performance
        expected_growth = last_price * (growth_potential + 1)

        return expected_growth

    def update_diversification_preferences(self, sectors_data, day):
        # favor volatile sectors more over month long time horizon
        total = 0
        sector_ratios = {}

        look_back = max(day - 30, 0)

        for sector in sectors_data:
            sector_performance = sector[3][look_back:day].copy()
            avg = sum(sector_performance)/len(sector_performance)

            if avg <= 0:
                ratio = 0
            else:
                sector_performance = np.asanyarray(sector_performance, dtype='object')
                std = np.std(sector_performance)
                ratio = std / avg
            total = total + ratio
            sector_ratios[sector[0]] = ratio

        for sector in sector_ratios:
            if total == 0:
                break
            sector_ratios[sector] = sector_ratios[sector] / total

        self.diversification_preferences = sector_ratios

    # evaluate performance in short term, only prices from last week
    def evaluate_sector_performance(self, stock, sector_performance, day):

        stock_performance = stock.get_share_price_history().copy()
        sector_performance = sector_performance.copy()

        # only consider values from the last month, myopic analysis
        look_back = max(day - 30, 0)

        stock_performance = stock_performance[look_back: day]
        sector_performance = sector_performance[look_back: day]

        # compare historical averages
        if len(stock_performance) == 0 or len(sector_performance) == 0:
            return 0

        stock_average_performance = sum(stock_performance) / len(stock_performance)
        sector_average_performance = sum(sector_performance) / len(sector_performance)
        if stock_average_performance == 0 or sector_average_performance == 0:
            return 0
        stock_day = min(day, len(stock_performance))
        sector_day = min(day, len(sector_performance))

        stock_growth_potential = (stock_performance[
                                      stock_day - 1] - stock_average_performance) / stock_average_performance
        sector_growth_potential = (sector_performance[
                                       sector_day - 1] - sector_average_performance) / sector_average_performance

        if stock_growth_potential == 0 or sector_growth_potential == 0:
            return 0

        growth_differential = (stock_growth_potential - sector_growth_potential) / sector_growth_potential
        return growth_differential

    def get_speculation(self, day):
        # used past returns to determine confidence level, should also effect risk
        # risk preference and speculation increase with past positive returns
        # should consider direction and magnitude of past returns, in a ratio of positive to negative
        # initial returns 0, speculation is 0
        # sees past month unweighted, myopic about rest
        if len(self.returns_history) == 0 or day == 0:
            return 0

        avg_return = np.average(self.returns_history)
        weighted_sum = 0
        pct_positive = 0
        for i in range(len(self.returns_history)):
            if self.returns_history[i] == 0:
                continue

            if self.returns_history[i] > 0:
                pct_positive += 1

            value = (self.returns_history[i] - avg_return) / self.returns_history[i]

            if day - i <= 30:
                weighted_value = value

            else:
                weighted_value = i / day * value

            weighted_sum = weighted_sum + weighted_value

        pct_positive = pct_positive / len(self.returns_history)

        speculation = weighted_sum * pct_positive
        return speculation

    def get_max_willingness_to_pay(self, stock, day, sector_performance):
        # expected future value of the stock over
        # value in comparison other stocks in sector - performs higher than average, want to buy more
        sector = self.evaluate_sector_performance(stock, sector_performance, day)
        budget = (self.get_budget() - self.initial_budget) / self.initial_budget
        risk = self.evaluate_risk(stock)
        estimated_earnings = self.estimate_future_earnings(stock, day)
        distribution = self.distribution_preference(stock.get_sector())
        speculation = self.get_speculation(day)

        max_WTP = min(
            ((estimated_earnings * (1 + speculation)) / risk) * distribution * (1 + budget) +
            (1 + sector) * stock.get_share_price()
            , self.budget)
        max_WTP = max(round(max_WTP, ndigits=2), 0)

        return max_WTP


# DAY TRADER CLASS
class Day_Trader(Trader):

    def __init__(self, budget):
        super().__init__(budget=budget)
        self.strategy = "Day Trader"
        print(self.strategy)
        # trader preferences
        # chases volatility, trades frequently, highly speculative
        self.risk_preference = random.uniform(.7, 1)
        self.trade_frequency = 364
        # diversification preferences not uniform, set according to which industry performs the best/ is most volatile

    def get_max_willingness_to_pay(self, stock, day, sector_performance):
        # expected future value of the stock over
        # value in comparison other stocks in sector - performs higher than average, want to buy more
        sector = self.evaluate_sector_performance(stock, sector_performance, day)
        budget = (self.get_budget() - self.initial_budget) / self.initial_budget
        risk = self.evaluate_risk(stock)
        estimated_earnings = self.estimate_future_earnings(stock, day)
        distribution = self.distribution_preference(stock.get_sector())
        speculation = self.get_speculation(day)

        max_WTP = min(
            ((estimated_earnings * (1 + speculation)) / risk) * distribution * (1 + budget) +
            (1 + sector) * stock.get_share_price()
            , self.budget)
        max_WTP = max(round(max_WTP, ndigits=2), 0)

        return max_WTP

    def update_diversification_preferences(self, sectors_data, day):
        # favor volatile sectors more

        total = 0
        sector_ratios = {}

        look_back = max(day - 7, 0)

        for sector in sectors_data:
            sector_performance = sector[3][look_back:day].copy()
            avg = sum(sector_performance)/len(sector_performance)
            if avg == 0:
                ratio = 0
            else:
                sector_performance = np.asarray(sector_performance, dtype='object')
                std = np.std(sector_performance)

                ratio = std / avg
            total = total + ratio
            sector_ratios[sector[0]] = ratio

        for sector in sector_ratios:
            if total == 0:
                break
            sector_ratios[sector] = sector_ratios[sector] / total

        self.diversification_preferences = sector_ratios

    # speculation based on past performance of trader
    def get_speculation(self, day):
        # used past returns to determine confidence level, should also effect risk
        # risk preference and speculation increase with past positive returns
        # should consider direction and magnitude of past returns, in a ratio of positive to negative
        # initial returns 0, speculation is 0
        if len(self.returns_history) == 0 or day == 0:
            return 0

        avg_return = np.average(self.returns_history)
        weighted_sum = 0
        pct_positive = 0
        for i in range(len(self.returns_history)):
            if self.returns_history[i] == 0:
                continue

            if self.returns_history[i] > 0:
                pct_positive += 1

            value = (self.returns_history[i] - avg_return) / self.returns_history[i]
            weighted_value = i / day * value
            weighted_sum = weighted_sum + weighted_value

        pct_positive = pct_positive / len(self.returns_history)

        speculation = weighted_sum * pct_positive
        return speculation

    # evaluate future earnings in short time horizon
    def estimate_future_earnings(self, stock, day):
        performance = stock.get_share_price_history().copy()
        look_back = max(day - 7, 0)

        performance = performance[look_back: day]
        average_performance = np.average(performance)
        last_price = stock.get_share_price()
        growth_potential = (last_price - average_performance) / average_performance
        expected_growth = last_price * (growth_potential + 1)

        return expected_growth

    # evaluate performance in short term, only prices from last week
    def evaluate_sector_performance(self, stock, sector_performance, day):
        stock_performance = stock.get_share_price_history().copy()
        sector_performance = sector_performance.copy()

        # only consider values from the last week, myopic analysis
        look_back = max(day - 7, 0)

        stock_performance = stock_performance[look_back: day]
        sector_performance = sector_performance[look_back: day]

        # compare historical averages
        if len(stock_performance) == 0 or len(sector_performance) == 0:
            return 0

        stock_average_performance = sum(stock_performance) / len(stock_performance)
        sector_average_performance = sum(sector_performance) / len(sector_performance)
        if stock_average_performance == 0 or sector_average_performance == 0:
            return 0
        stock_day = min(day, len(stock_performance))
        sector_day = min(day, len(sector_performance))

        stock_growth_potential = (stock_performance[
                                      stock_day - 1] - stock_average_performance) / stock_average_performance
        sector_growth_potential = (sector_performance[
                                       sector_day - 1] - sector_average_performance) / sector_average_performance

        if stock_growth_potential == 0 or sector_growth_potential == 0:
            return 0

        growth_differential = (stock_growth_potential - sector_growth_potential) / sector_growth_potential
        return growth_differential


# PESSIMISTIC TRADER CLASS
class Pessimistic(Trader):

    def __init__(self, budget):
        super().__init__(budget=budget)
        self.strategy = "Pessimistic Trader"
        # trader preferences
        # low risk preference, trades a below normal amount, reacts very negatively to losses that affects speculation,
        # limits budget when faced with negative losses, weighs losses greater than gains
        # updates diversification preferences away from negatively performing industries
        self.risk_preference = random.uniform(.01, .25)
        self.trade_frequency = 30
        self.speculation = 0
        self.pessimism = random.uniform(1, 3)

    def get_max_willingness_to_pay(self, stock, day, sector_performance):
        # expected future value of the stock over
        # value in comparison other stocks in sector - performs higher than average, want to buy more
        sector = self.evaluate_sector_performance(stock, sector_performance, day)
        budget = self.get_trading_budget(day)
        # budget = (self.get_budget() - self.initial_budget) / self.initial_budget
        risk = self.evaluate_risk(stock)
        estimated_earnings = self.estimate_future_earnings(stock)
        distribution = self.distribution_preference(stock.get_sector())
        speculation = self.get_speculation(day)

        max_WTP = min(
            ((estimated_earnings * (1 + speculation)) / risk) * distribution * (1 + budget) +
            (1 + sector) * stock.get_share_price()
            , self.budget)
        max_WTP = max(round(max_WTP, ndigits=2), 0)

        return max_WTP

    def update_diversification_preferences(self, sectors_data, day):
        # adverse to volatile sectors
        total = 0
        sector_ratios = {}
        for sector in sectors_data:
            sector_performance = sector[3].copy()
            last_value = sector_performance[day - 1]
            if last_value == 0:
                ratio = 0
            else:
                sector_performance = np.asanyarray(sector_performance, dtype='object')
                week_52_low = np.amin(sector_performance)
                pessimism = (last_value - week_52_low) / last_value

                std = np.std(sector_performance)
                avg = np.average(sector_performance)

                ratio = (1 - std / avg) * pessimism
                total = total + ratio

            sector_ratios[sector[0]] = ratio

        for sector in sector_ratios:
            if total == 0:
                break
            sector_ratios[sector] = sector_ratios[sector] / total

        self.diversification_preferences = sector_ratios

    def get_trading_budget(self, day):
        # budget * confidence multiplier, loss aversion
        new_budget = self.get_budget()

        if len(self.returns_history) == 0 or day == 0:
            return new_budget

        avg_return = np.average(self.returns_history)
        weighted_sum = 0
        pct_negative = 0
        for i in range(len(self.returns_history)):
            weight = 1
            if self.returns_history[i] == 0:
                continue

            if self.returns_history[i] < 0:
                pct_negative += 1
                # loss are magnified, remain in memory
                weight = self.pessimism

            value = (self.returns_history[i] - avg_return) / self.returns_history[i]
            weighted_value = weight * i / day * value
            weighted_sum = weighted_sum + weighted_value

        pct_negative = pct_negative / len(self.returns_history)

        # larger weighted_sum means more high magnitude positive returns, weigh pct_negative less
        confidence_loss = (1 - weighted_sum) * pct_negative
        new_budget = (1 - confidence_loss) * new_budget

        return new_budget

    def get_speculation(self, day):
        # used past returns to determine confidence level, should also effect risk
        # risk preference already low
        # negative returns weighed more heavily, according to pessimism level
        # should consider direction and magnitude of past returns, in a ratio of positive to negative
        # initial returns 0, speculation is 0
        if len(self.returns_history) == 0 or day == 0:
            return 0

        avg_return = np.average(self.returns_history)
        weighted_sum = 0
        pct_negative = 0
        for i in range(len(self.returns_history)):
            weight = 1
            if self.returns_history[i] == 0:
                continue

            if self.returns_history[i] < 0:
                pct_negative += 1
                # loss are magnified, remain in memory
                weight = self.pessimism

            value = (self.returns_history[i] - avg_return) / self.returns_history[i]
            weighted_value = weight * i / day * value
            weighted_sum = weighted_sum + weighted_value

        pct_negative = (pct_negative / len(self.returns_history)) * self.pessimism

        pct_positive = 1 - pct_negative

        speculation = weighted_sum * pct_positive
        return speculation


# BAYESIAN TRADER CLASS
class Bayesian(Trader):

    def __init__(self, budget):
        super().__init__(budget=budget)
        self.strategy = "Bayesian Trader"
        # trader preferences
        # trader estimates future earnings based on bayes theorem
        self.risk_preference = random.uniform(.01, 1)
        self.trade_frequency = 52
        self.speculation = 0

    # uses Bayes rule to get growth potential
    def estimate_future_earnings(self, stock):
        performance = stock.get_share_price_history()
        last_price = performance[-1]

        # Bayes Theorem = P(FP>CP|CP>PP) = [P(FP>CP) * P(CP>PP|FP>CP)]/ P(CP>PP)
        week_52_high = np.amax(performance)
        week_52_low = np.amin(performance)
        week_52_range = week_52_high - week_52_low
        cp = stock.get_share_price()
        if len(performance) > 2:
            pp = performance[-2]
        else:
            pp = performance[-1]
        A = (week_52_high - cp)/week_52_range
        B = (cp - week_52_low)/week_52_range
        B_A = (pp - week_52_low)/week_52_range

        growth_potential = round((A * B_A)/B, ndigits=2)
        expected_growth = last_price * (growth_potential + 1)

        return expected_growth

    def evaluate_sector_performance(self, stock, sector_performance):
        stock_performance = stock.get_share_price_history()
        # use Bayes rule to derive growth potential differential
        # Bayes Theorem = P(FP>CP|CP>PP) = [P(FP>CP) * P(CP>PP|FP>CP)]/ P(CP>PP)
        stock_week_52_high = np.amax(stock_performance)
        stock_week_52_low = np.amin(stock_performance)
        stock_week_52_range = stock_week_52_high - stock_week_52_low
        stock_cp = stock.get_share_price()
        if len(stock_performance) > 2:
            stock_pp = stock_performance[-2]
        else:
            stock_pp = stock_performance[-1]
        stock_A = (stock_week_52_high - stock_cp)/stock_week_52_range
        stock_B = max((stock_cp - stock_week_52_low)/stock_week_52_range, .01)
        stock_B_A = (stock_pp - stock_week_52_low)/stock_week_52_range
        stock_growth_potential = round((stock_A * stock_B_A)/stock_B, ndigits=2)

        sector_week_52_high = np.amax(sector_performance)
        sector_week_52_low = np.amin(sector_performance)
        sector_week_52_range = sector_week_52_high - sector_week_52_low
        sector_cp = sector_performance[-1]
        sector_pp = sector_performance[-2]
        sector_A = (sector_week_52_high - sector_cp)/sector_week_52_range
        sector_B = max((sector_cp - sector_week_52_low)/sector_week_52_range, .01)
        sector_B_A = (sector_pp - sector_week_52_low)/sector_week_52_range

        sector_growth_potential = round((sector_A * sector_B_A)/sector_B, ndigits=2)

        if stock_growth_potential == 0:
            return 0

        growth_differential = (stock_growth_potential - sector_growth_potential) / sector_growth_potential
        return growth_differential

    def get_max_willingness_to_pay(self, stock, day, sector_performance):
        # expected future value of the stock over
        # value in comparison other stocks in sector - performs higher than average, want to buy more
        # estimate earnings with bayes rule
        # evaluate sector with bayes rule
        sector = self.evaluate_sector_performance(stock, sector_performance)
        budget = (self.get_budget() - self.initial_budget) / self.initial_budget
        risk = self.evaluate_risk(stock)
        estimated_earnings = self.estimate_future_earnings(stock)
        distribution = self.distribution_preference(stock.get_sector())
        speculation = self.get_speculation(day)

        max_WTP = min(
            ((estimated_earnings * (1 + speculation)) / risk) * distribution * (1 + budget) +
            (1 + sector) * stock.get_share_price()
            , self.budget)
        max_WTP = max(round(max_WTP, ndigits=2), 0)

        return max_WTP
