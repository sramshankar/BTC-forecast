import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

btc = pd.read_csv("main.csv")
btc.head()
 
btc = btc[["Close"]]
prices = btc.to_numpy()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_prices = scaler.fit_transform(prices)

X_train = []
y_train = []
for i in range(200,188317):
    X_train.append(scaled_prices[i-200:i,0])
    y_train.append(scaled_prices[i,0])
X_train,y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units =50))
model.add(Dropout(0.2))
model.add(Dense(units =1))
model.compile(optimizer ='adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 3, batch_size = 32)

train_pred = model.predict(x_train).flatten()
train_results = pd.DataFrame(data = {'Train Pred.':train_pred, 'Actual':y_train})
import matplotlib.pyplot as plt
plt.plot(train_results['Train Pred.'])
plt.plot(train_results['Actual'])

