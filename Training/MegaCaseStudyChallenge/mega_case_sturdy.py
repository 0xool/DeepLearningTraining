
# Part 1 


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Applications.csv')
#%%
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# %%
# Feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
x = sc.fit_transform(x)

# %%
from minisom import MiniSom 

# %%
som = MiniSom(x=10,y=10,input_len=15,sigma = 1.0, learning_rate = 0.5)

# %%
som.random_weights_init(x)
som.train_random(data=x,num_iteration=100)


# %%

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,X in enumerate(x) :
    w = som.winner(X)
    plot( w[0] + 0.5, w[1] + 0.5,markers[y[i]],markeredgecolor = colors[y[i]] , markerfacecolor = 'None',markersize = 10, markeredgewidth = 2)

show()


# %%
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(5,7)], mappings[(5,4)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# %%

# part 2 supervised section
customers = dataset.iloc[:,1:].values


# %%
is_fraud = np.zeros(len(dataset))

# %%
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds :
        is_fraud[i] = 1


# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

#%%
# Make ANN
# Importing Keras Library
import keras
from keras.models import Sequential
from keras.layers import Dense
#%%
from keras.layers import Dropout

#%%
# Initialize ANN
classifier = Sequential()


# Adding first input and hiddent layer 
classifier.add(Dense(units=2,kernel_initializer='uniform',activation='relu' ,input_shape=(15,)))
# Adding another hidden layer

# Adding output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))


# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(x=customers,y=is_fraud,batch_size=1,epochs=2)

#%%
# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Probebility of Frauds
y_pred = classifier.predict(customers)

# %%
y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred) , axis = 1)
#%%
y_pred = y_pred[y_pred[:,1].argsort()]
# %%
