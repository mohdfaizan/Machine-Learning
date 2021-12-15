# Hybrid deep Learning model combination of Supervised and Unsupervised 

import os
os.chdir(r'D:\computer vision')

# Unsupervised Model with Self Organizing Maps (Identifying the Fraud)
import pandas as pd
import numpy as np

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X  = sc.fit_transform(X)

# training SOM
from minisom import MiniSom
som = MiniSom(10,10,input_len=15,sigma=1.0)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

# visualizing
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor=colors[y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
show()

# finding the frauds
mappings = som.win_map(X)
# (8,1) and (6,8) is the cordinate got from the above plot 
frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]))
frauds = sc.inverse_transform((frauds))

# Supervised Machine learning

#creating Matrix of Features
customers = dataset.iloc[:,1:].values
#creating Dependent variable
is_fraud = np.zeros((len(dataset)))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)
import tensorflow as tf
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=2, activation='relu', input_dim=15))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(customers, is_fraud, batch_size = 1, epochs = 10)
y_pred = ann.predict(customers)

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]

print(y_pred)
