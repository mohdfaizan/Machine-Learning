import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing te dataset
import os
os.chdir(r'D:\computer vision')
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = dataset_train.iloc[:,1:2].values

# Feature scaling the dataaset
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
x_train_data = ss.fit_transform(train_set)

# creating a datastructure with 60 timesteps and 1 output
# that means it loocks back for 60 pattern
x_train = []
y_train = []
for i in range(60,1258):
    x_train.append(x_train_data[i-60:i,0])
    y_train.append(x_train_data[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)

#reshaping
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

# RNN Construction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# make the value of return_sequences=False in the last LSTM Layer, its default value is False
# therefore i removed it.
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train,y_train,epochs=100,batch_size=32)

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs  = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = ss.transform(inputs)

x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

y_pred = regressor.predict(x_test)

y_pred = ss.inverse_transform(y_pred)

plt.plot(test_set,color='red',label="real google stock price")
plt.plot(y_pred,color='blue',label="predicted google stock price")
plt.title("Google stock Price prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()




