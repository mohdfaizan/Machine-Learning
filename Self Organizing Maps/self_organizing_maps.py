import pandas as pd
import numpy as np

import os
os.chdir(r'D:\computer vision')
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
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]))
frauds = sc.inverse_transform((frauds))
