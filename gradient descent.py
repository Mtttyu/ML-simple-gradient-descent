# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def MSE(x,y,weight):
    inner = np.power(((X * weight.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def MAE(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))


def gradientDescent(X, y, weight, alpha, iters):
    temp = np.matrix(np.zeros(weight.shape))
    parameters = int(weight.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * weight.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = weight[0,j] - ((alpha / len(X)) * np.sum(term))
            
        weight = temp
        cost[i] = MSE(X, y, weight)
        
    return weight, cost



label = []
feature = [[],[],[]] 
for i in range(1000):
    X1 = random.randint(0,100)
    feature[0].append(X1)
    
    X2 = random.randint(0,200)
    feature[1].append(X2)
    
    X3 = random.randint(0,300)
    feature[2].append(X3)
    
    Y = 5 * X1 + 3 * X2 + 1.5 * X3 + 6
    label.append(Y)
    
    
    
data = {'X1': feature[0], 'X2': feature[1], 'X3': feature[2], 'y': label}  
df = pd.DataFrame(data)
# rescaling data
df = (df - df.mean()) / df.std()

print(df)    
cols = df.shape[1]
X = df.iloc[:,0:cols-1]
y = df.iloc[:,cols-1:cols]


X = np.matrix(X.values)
y = np.matrix(y.values)
weights = np.matrix(np.array([0,0,0]))
    

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.33)

# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

# perform linear regression on the data set
weights, cost = gradientDescent(X, y, weights, alpha, iters)

y_pred= np.matrix(X * weights.T)

# get the cost (error) of the model
cost1 = MSE(X, y, weights)
cost2=MAE(y,y_pred)
print('**************************************')
print('weights = ' , weights)
print('**************************************')
#print('cost2  = ' , cost2[0:50] )
print('mean squared error = ' , cost1)
print('**************************************')
print('mean abs error = ' , cost2)











