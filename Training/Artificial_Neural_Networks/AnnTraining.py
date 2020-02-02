
#%% 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#%%
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
#%%
# Male/Female
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X = X[: , 1:]

#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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

#%%
# Adding first input and hiddent layer 
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu' ,input_shape=(11,)))
classifier.add(Dropout(rate=0.1))
# Adding another hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.1))
# Adding output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

#%%
# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#%%
# Fitting the ANN to the Training set
classifier.fit(x=X_train,y=y_train,batch_size=10,epochs=100)

#%%
# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#%%
# Predict a Single customer

new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,5000]])))
#%%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# %%
# Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier () :
    classifierB = Sequential()
    classifierB.add(Dense(units=6,kernel_initializer='uniform',activation='relu' ,input_shape=(11,)))
    classifierB.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    classifierB.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    classifierB.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifierB

classifierD = KerasClassifier(build_fn = build_classifier,batch_size = 10 ,epochs = 100)
accuracies = cross_val_score(estimator = classifierD , X = X_train , y = y_train , cv = 10 , n_jobs = -1)

#%%
mean = accuracies.mean()

# %%
# Tuning ANN
from sklearn.model_selection import GridSearchCV

def build_classifier (optimize) :
    classifierB = Sequential()
    classifierB.add(Dense(units=6,kernel_initializer='uniform',activation='relu' ,input_shape=(11,)))
    classifierB.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    classifierB.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    classifierB.compile(optimizer= optimize,loss='binary_crossentropy',metrics=['accuracy'])
    return classifierB

classifierD = KerasClassifier(build_fn = build_classifier)
gridParameters = {'batch_size': [25,32],'epochs': [100,500],'optimize':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifierD,param_grid=gridParameters,scoring='accuracy',cv=10)
grid_search = grid_search.fit(X=X_train,y=y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# %%
