import numpy as np
import pandas as pd
from sklearn import datasets
# iris data


iris = datasets.load_iris()

X = iris.data[:,:2]
y = (iris.target !=0)*1

X = np.hstack((np.ones((X.shape[0],1)),X))

theta = np.random.rand(X.shape[1])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def h(X, theta):
    return sigmoid(np.dot(X,theta))

def cost(X, y, theta):
    m = len(X)
    h_x = h(X,theta)
    J = (-1/m)*(np.dot(y,np.log(h_x))+np.dot((1-y), 
        np.log(1-h_x)))
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(X)
    J_history = []

    for i in range(num_iters):

        theta -= (alpha / m) * np.dot(X.T, h(X,theta)-y)

        J_history.append(cost(X,y,theta))
    return theta, J_history

alpha = 0.01
num_iters=10000


theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

print(theta)

