"""
Author: Jason Teng
"""

import pdb
import random
import pylab as pl
import numpy as np
from scipy.optimize import fmin_bfgs

# Question 1:

# guess is of form (xpos, ypos)
def minimize(f, g, guess, step_size, threshold):
    iterations = 0
    while True:
        iterations += 1
        # compute current gradient by passing in the current positions
        gradient = g(guess)
        # update position by using gradient and learning rate
        nextpos = list(guess).copy()
        for i in range(len(nextpos)):
            nextpos[i] -= step_size*gradient[i]
        # compute f(cur_x, cur_y) and f(update_x, update_y)
        if abs(f(nextpos)-f(guess)) < threshold:
            return nextpos, iterations
        else:
            guess = nextpos

def minimizeNumeric(f, guess, step_size, threshold):
    iterations = 0
    while True:
        iterations += 1
        # compute current gradient by passing in the current positions
        gradient = numericGradient(f, guess)
        # update position by using gradient and learning rate
        nextpos = list(guess).copy()
        for i in range(len(nextpos)):
            nextpos[i] -= step_size*gradient[i]
        # compute f(cur_x, cur_y) and f(update_x, update_y)
        if abs(f(nextpos)-f(guess)) < threshold:
            return nextpos, iterations
        else:
            guess = nextpos

def numericGradient(f, X):
    h = 0.0000005 # hard-coded constant h
    H = np.identity(len(X)) * h
    g = []
    for i in H:
        g.append((f(np.add(X, i)) - f(np.add(X, -i)))/(2*h))
    return g
 
# f1 is f(x, y) = 2x^2 + 4y^2
def f1(x):
     return 2*(x[0]**2) + 4*(x[1]**2)

# gradient of 2x^2 + 4y^2 is just 4x, 8y
def g1(x):
    return 4*x[0], 8*x[1]

# simple quadratic function for convex function
def f2(x):
    return -5 + x[0]**2

def g2(x):
    return [2*x[0]]

# function with 2 minima
def f3(x):
    return 5 + (x[0]+2)*(x[0]-1)*(5*x[0]-4)*(x[0]-6)*(x[0]-7)

def g3(x):
    return [25*(x[0]**4)-256*(x[0]**3)+549*(x[0]**2)+464*x[0]-692]

print(minimize(f3, g3, [0], 0.001, 0.000001))
print(minimizeNumeric(f3, [0], 0.001, 0.000001))

print(fmin_bfgs(f3,[0],))


# Question 2:

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function
    
    # constuct the design matrix, the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    w = regressionFit(Y, phi)

    print('w', w)
    print('SSE: ', SSE(Y, phi, w))
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])

def designMatrix(X, order):
    dm = []
    for x in X:
        row = [1]
        for i in range(order):
            row.append(x[0]**(i+1))
        dm.append(row)
    return np.asarray(dm)

def regressionFit(Y, phi):
    return np.linalg.pinv(phi.T.dot(phi)).dot(phi.T).dot(Y)
    
def SSE(Y, phi, w):
    sum = 0
    for i in range(len(phi)):
        sum += (Y[i] - w.T.dot(phi[i]))**2
    return sum

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

# Question 3

def ridgeRegressionPlot(X, Y, order, l):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function
    
    # constuct the design matrix, the 0th column is just 1s.
    phi = designMatrix(X, order)
    
    I = np.identity(order+1)
    # compute the weight vector
    w = ridgeRegressionFit(Y, phi, l, I)

    print('w', w)
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])

def ridgeRegressionFit(Y, phi, l, I):
    return np.linalg.pinv(np.add(np.dot(l, I), (phi.T.dot(phi)))).dot(phi.T).dot(Y)

def evaluate(X, Y, order, l, vdata):
    pl.plot(vdata[0].T.tolist()[0],vdata[1].T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function
    
    # constuct the design matrix, the 0th column is just 1s.
    phi = designMatrix(X, order)
    
    I = np.identity(order+1)
    # compute the weight vector
    w = ridgeRegressionFit(Y, phi, l, I)

    print('w', w)
    print('SSE: ', SSE(Y, phi, w))
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])

"""
data = bishopCurveData()
regressionPlot(data[0], data[1], 0)
pl.show()
regressionPlot(data[0], data[1], 1)
pl.show()
regressionPlot(data[0], data[1], 3)
pl.show()
regressionPlot(data[0], data[1], 9)
pl.show()
ridgeRegressionPlot(data[0], data[1], 3, 0.001)
pl.show()
ridgeRegressionPlot(data[0], data[1], 3, 0.01)
pl.show()
ridgeRegressionPlot(data[0], data[1], 3, 0.1)
pl.show()
ridgeRegressionPlot(data[0], data[1], 3, 1)
pl.show()
ridgeRegressionPlot(data[0], data[1], 9, 0.001)
pl.show()
ridgeRegressionPlot(data[0], data[1], 9, 0.01)
pl.show()
ridgeRegressionPlot(data[0], data[1], 9, 0.1)
pl.show()
ridgeRegressionPlot(data[0], data[1], 9, 1)
pl.show()
ridgeRegressionPlot(data[0], data[1], 7, 0.001)
pl.show()
ridgeRegressionPlot(data[0], data[1], 7, 0.01)
pl.show()
ridgeRegressionPlot(data[0], data[1], 7, 0.1)
pl.show()
ridgeRegressionPlot(data[0], data[1], 7, 1)
"""
adata = regressAData()
bdata = regressBData()
vdata = validateData()
ridgeRegressionPlot(adata[0], adata[1], 5, 0.001)
pl.show()
evaluate(adata[0], adata[1], 5, 0.001, vdata)
pl.show()
ridgeRegressionPlot(adata[0], adata[1], 3, 0.001)
pl.show()
evaluate(adata[0], adata[1], 3, 0.001, vdata)
pl.show()
ridgeRegressionPlot(bdata[0], bdata[1], 5, 1)
pl.show()
evaluate(bdata[0], bdata[1], 5, 1, vdata)
pl.show()
ridgeRegressionPlot(bdata[0], bdata[1], 3, 1)
pl.show()
evaluate(bdata[0], bdata[1], 3, 1, vdata)















