"""Linear regression from scratch

    X: input value (1 variable)
    y: output value (1 variable)
    theta: parameters
    learning rate: 0.01
    iterations: 1000
    m = num of rows

    Things to include

    -hypothesis function
    -cost function
    -gradient descent
    -prediction

"""
import numpy as np

class LinearRegression():
    """
    model = LinearRegression()
    model.fit(X,y)
    model.predict(X) 

    """
    def __init__(self, X, y, m, iters, lr,theta):


        self.X = X
        self.y = y
        self.m = m
        self.iters = iters
        self.lr = lr
        self.theta = theta
    
    # define the hypothesis function

    def hypothesis(self,X,theta):
        # h(x) = X*W + B
        y_pred = np.dot(self.X,self.theta)
        return y_pred

    """Cost Function: It is basically the error function that measures the distance
    between true and predicted value"""

    def cost(self,X,y,m):

        y_pred = self.hypothesis(self.X,self.theta)
        Jcost = (1./2*m)*np.sum((np.square(y_pred-y)))
        return Jcost

    def gradient_descent(self, X, y,iters,lr,theta):

        costs = []

        for i in range(iters):

            y_pred = self.hypothesis(self.X, self.theta)

            theta -= 1/m*lr*np.sum(np.dot(X.T,(y_pred-y)))

            
            costs.append(theta)

            return theta

    def predict(self, X):

        theta = self.gradient_descent(self.X,self.y,self.iters,self.lr,self.theta)

        return self.hypothesis(self.X,self.theta)


# define the datasets before feeding
np.random.seed(0)
X = 2*np.random.randn(100,1)
y = 4*3*X+np.random.randn(100,1)

m = len(X)

x_ = np.ones((m,1))
x__ = X.reshape(m,1)

X = np.hstack((x_,x__))

theta = np.zeros((X.shape[1],1))

iters=1000
lr=0.01
model = LinearRegression(X,y,m,iters,lr,theta)

print(model.predict(X))


