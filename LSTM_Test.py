import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.metrics import acc
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

look_back = 100
forward_days = 30
# num_periods = 10

# load and format data-frame
stock_name = "TSLA"

directory = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(directory + "/Stock Data/Test_data/TSLA.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df['Close']

# data normalization
data_array = df.values.reshape(df.shape[0], 1)
scl = MinMaxScaler()
data_array = scl.fit_transform(data_array)


# split into train and test data
print("length of data: " + str(len(data_array)))
# division = len(data_array) - num_periods*forward_days
division = math.floor(0.4*len(data_array))

print("division point: " + str(division))

test_data = data_array[division-look_back:]
train_data = data_array[:division]


# Get the data and splits in input X and output Y, by splitting in `n` past days as input X
# and `m` coming days as Y.
# splits data into [20, 30, 40] -> [50]
def processData(data, look_back, forward_days, jump=1):
    X, Y = [], []
    for i in range(0, len(data) - look_back - forward_days + 1, jump):
        X.append(data[i:(i+look_back)])
        Y.append(data[(i+look_back):(i+look_back+forward_days)])
    return np.array(X), np.array(Y)


# prepare test and train data
X_test, y_test = processData(test_data, look_back, forward_days, forward_days)
y_test = np.array([list(a.ravel()) for a in y_test])
# print((y_test))

X, y = processData(train_data, look_back, forward_days)

# testsize = round(look_back/len(data_array), 2)

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=42)

# Model parameters
NUM_NEURONS_FirstLayer = 128
NUM_NEURONS_SecondLayer = 64
EPOCHS = 5

# Build the model
model = Sequential()
model.add(LSTM(NUM_NEURONS_FirstLayer, input_shape=(look_back,1), return_sequences=True))
model.add(LSTM(NUM_NEURONS_SecondLayer, input_shape=(NUM_NEURONS_FirstLayer, 1)))
model.add(Dense(forward_days))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
model.compile(loss='mean_squared_error', optimizer='adam')

# check data validation and loss
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_validate, y_validate), shuffle=True,
                    batch_size=2, verbose=2)

# Data visualization
plt.figure(figsize=(5, 5))

# Plot validation loss
# plt.plot(df['Date'], df['Close'], label = 'Tesla')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.show()

# plot accuracy during training
# pyplot.subplot(212)
# pyplot.title('Accuracy')
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
# pyplot.legend()
# pyplot.show()

# Predict test set to see results, step wise direction results

# what does this do?
leftover = division % forward_days+1

# process training data
# split the data up into train and test segments
X_train, y_train = processData(train_data, look_back, forward_days, forward_days)
X_test, y_test = processData(test_data, look_back, forward_days, forward_days)

# predict on training model
X_train = model.predict(X_train)
X_train = X_train.ravel()

# predict on testing model
X_test = model.predict(X_test, verbose=0)
X_test = X_test.ravel()

y = np.concatenate((y_train, y_test), axis=0)

# revert data
X_train = scl.inverse_transform(X_train.reshape(-1, 1))
X_test = scl.inverse_transform(X_test.reshape(-1, 1))
y = scl.inverse_transform(y.reshape(-1, 1))


print(X_test)
print("number of days forecasted: " + str(len(X_test)))
print("number of training days: " + str(len(X_train)))

plt.figure(figsize = (5, 5))

# Data in Train/Validation
plt.plot([x for x in range(look_back+leftover, len(X_train)+look_back+leftover)], X_train, color='r', label='Train')
# Data in Test
plt.plot([x for x in range(look_back +leftover+ len(X_train), len(X_train)+len(X_test)+look_back+leftover)], X_test, color='y', label='Test')
# True data
plt.plot([x for x in range(look_back+leftover, look_back+leftover+len(X_train)+len(X_test))], y, color='b', label='Actual History')

plt.title(stock_name)
plt.xlabel("Days")
plt.ylabel("Price")

plt.legend(loc='best')
plt.show()

# np.save('/Users/rayhonchoudhry/Desktop/School Work/CompSci/Comps Project/Stock Data/TSLA_Prediction_data.npy',
# X_train)
# extrapolate

