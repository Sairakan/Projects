import copy
import pdb
import numpy as np
from matplotlib import pyplot as pl
from cvxopt import matrix, solvers

# X is data matrix (each row is a data point)
# Y is desired output (1 or -1)
# scoreFn is a function of a data point
# values is a list of values to plot

def plotDecisionBoundary(X, Y, scoreFn, values, title = ""):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    pl.figure()
    CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=(1-Y).flatten(), s=50, cmap = pl.cm.cool)
    pl.title(title)
    pl.axis('tight')
    pl.show()

#########################################################################
# 1 Logistic Regression

# parameters
dataname = 'nls'
#print '======Training======'
# load data from csv files
train = np.loadtxt('data/data_'+ dataname +'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.

def sigmoid(x):
    return 1. / (1+np.exp(-x))

def cost(X, Y, w, l):
    e = 0
    for i in range(len(Y)):
        e += ELR(X[i].dot(w), Y[i]) + l*(np.linalg.norm(w)**2)
    return e

def ELR(t, y):
    return np.log(1+np.exp(-t*y))

def costgrad(X, Y, w, l):
    grad = []
    for j in range(len(w)):
        pgrad = 2*l*np.linalg.norm(w)
        for i in range(len(X)):
            pgrad -= Y[i]*X[i][j]*sigmoid(Y[i]*(-X[i].dot(w)))
        grad.append(pgrad)
    return grad

# set initial weights to 0
w = []
for i in range(len(X[0])):
    w.append(1)

# perform regression
def logisticregression(X, Y, w, l, lr, iters):
    for j in range(iters):
        grad = costgrad(X, Y, w, l)
        for i in range(len(w)):
            w[i] -= lr*grad[i]
    return w

w = logisticregression(X,Y,w,1e-3,1e-2,1000)

# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    return sigmoid(x.dot(w))

def numerrors(f):
    errors = 0
    for i in range(len(Y)):
        if f(X[i]) >= 0.5 and Y[i] < 0:
            errors += 1
        elif f(X[i]) < 0.5 and Y[i] > 0:
            errors += 1
    return errors

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train: ' + dataname)

#print '======Validation======'
# load data from csv files
validate = np.loadtxt('data/data_'+ dataname +'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate: ' + dataname)

print('errors in logistic regression on ' + dataname + ' data: ' + str(numerrors(predictLR)))

# 1.3: Quadratic Basis Functions
dataname = 'nonlin'
train = np.loadtxt('data/data_'+ dataname +'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# extend X
x = []
for i in range(len(X)):
    r = [X[i][0], X[i][1], X[i][0]**2, X[i][1]**2, X[i][0]*X[i][1]]
    x.append(r)
X = np.array(x)

w = []
for i in range(len(X[0])):
    w.append(1)

w = logisticregression(X,Y,w,1e-1,1e-3,1000)

def predictLRQ(x):
    x = [x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1]]
    return sigmoid(np.dot(x, w))



plotDecisionBoundary(X,Y,predictLRQ,[0.5],title='LRQ Train: ' + dataname)

# load data from csv files
validate = np.loadtxt('data/data_'+ dataname +'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLRQ, [0.5], title = 'LRQ Validate: ' + dataname)

print('errors in logistic regression on ' + dataname + ' data: ' + str(numerrors(predictLRQ)))