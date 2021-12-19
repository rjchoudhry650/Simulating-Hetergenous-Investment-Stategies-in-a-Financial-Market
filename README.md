# Stock_Market_Simulation
This is a simulation of a financial market for the purposes of testing heterogeneous investment strategies via agent based simulation. 
Price variation is derived in a combination of stochastic and LSTM extrapolation from historical price data. The simulation looks at the performance of 5 trader archetypes: Rational, Day Trader, Trend Investor, Bayesian Trader, and Pessimistic. These traders make unique buying and selling decisions over a basket of 500 extrapolated price histories, selected from a possible lot of 2800 extrapolations. Prices functions as a combinations of LSTM derived extrapolation from real world price histories, and stochastic variation in the form of individual share price variation and exogenous market shocks, i.e. economic events.

## How To Run
The simulation is ran from Stock_Market_Simulation.py. It uses class files Event.py, Stock.py and Trader.py, and utilizes data from the Stock Data folder.

### Dependencies
You will need the dependencies listed below: Note: it is encouraged that you utilize a venv through either pip or anaconda

- python3
- tensorflow
- pandas
- pandas-datareader
- keras
- sklearn
- matplotlib
- numpy

### Install
```
$ git clone https://github.com/rjchoudhry650/Stock_Market_Simulation.git

$ cd Stock_Market_Simulation

$ pip3 install -r Requirements.txt
```

### Running
Once all nececessary packages have been installed, execute simulation via: 
```python Stock_Market_Simulation.py```

The code will prompt the use for the number of traders, number of days, and number of stocks. The maximum acceptable value for each is listed in the prompt.
Larger numbers of traders and stocks will require time complete. For a basic demo, it is suggested 5 traders in a 20 market for any number of days less than 365.

## Replication
In the spirit of full disclosure, Compile_Data.py has been included to show how to the price data was collected and extrapolated from for the purposes of running the simulation. Please note that running this file will take anywhere from 30-45 hours; the data is already available in the Stock Data/Extrapolation folder. 

```python Compile_Data.py```

In addition, LSTM_Test.py is provided in order to show how the LSTM model was used to generate the extrapolated values; you are welcome to run this to observe the methodology, doing so should take around 2-5 minutes.

```python LSTM_Test.py```

