#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

# %%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_norm = sc.fit_transform(training_set)

# %%
X_train = []
Y_train = []
for i in range(60,1258):
    X_train.append(training_set_norm[i - 60:i,0])
    Y_train.append(training_set_norm[i,0])
X_train,Y_train = np.array(X_train),np.array(Y_train)

# %%
# Reshaping
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# %%
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
# %%
regressor = Sequential()
regressor.add(LSTM(units=100,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# %%
regressor.add(LSTM(units=100,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=100,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=100,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=100))
regressor.add(Dropout(0.2))

# %%
regressor.add(Dense(units=1))

# %%
regressor.compile(optimizer= 'adam',loss='mean_squared_error')

# %%
regressor.fit(x=X_train,y=Y_train,batch_size=32,epochs=100)

# %%
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)

inputs = sc.fit_transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i - 60:i,0])
X_test = np.array(X_test)

# %%
# Reshaping
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predictions = regressor.predict(X_test)
predictions = sc.inverse_transform(predictions)



# %%
plt.plot(test_set,color = 'red',label='real google stock price')
plt.plot(predictions,color = 'blue',label='predicted google stock price')
plt.title('google stock price prediction')
plt.xlabel('time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
 

# %%
