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

###################################################################
# 2 SVM testing

# parameters
dataname = 'ls'
#print '======Training======'
# load data from csv files
train = np.loadtxt('data/data_'+ dataname +'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Carry out training, primal and/or dual
# primal SVM
# regularization term
C = 1/(2*1e-3)
# initialize P
P = np.zeros([1+len(X[0])+len(Y), 1+len(X[0])+len(Y)])
for i in range(1,len(X[0])+1):
    P[i][i] = 1
P = matrix(P)
    
# initialize q
q = []
for i in range(1+len(X[0])):
    q.append([0.])
for i in range(len(Y)):
    q.append([C])
q = matrix(q).T

# initialize G
G0 = []
G1 = []
G2 = np.zeros([len(Y),len(Y)])
G3 = np.zeros([len(Y),len(Y)]) 
for i in range(len(Y)):
    r0 = [Y[i][0]]
    r1 = [0]
    for j in range(len(X[0])):
        r0.append(Y[i][0]*X[i][j])
        r1.append(0)
    G0.append(r0)
    G1.append(r1)
    G2[i][i] = -1
    G3[i][i] = -1
for i in range(len(G1)):
    G0[i].extend(G2[i])
    G1[i].extend(G3[i])
    G0.append(G1[i])
G = matrix(G0).T

# initialize H
h = []
for i in range(len(Y)):
    h.append([-1.])
for i in range(len(Y)):
    h.append([0])
h = matrix(h).T
    
# find the solution
solution = solvers.qp(P,q,G,h)
xvals = np.array(solution['x'])

def predictSVM(x):
    return xvals[0] + x.dot(xvals[1:3])

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'Primal SVM Train: ' + dataname)


#print '======Validation======'
# load data from csv files
validate = np.loadtxt('data/data_'+ dataname +'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'Primal SVM Validate: ' + dataname)

# Dual SVM

train = np.loadtxt('data/data_'+ dataname +'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# regularization term
C = 1/(2*1e-3)
# initialize P
P = []
for i in range(len(Y)):
    r = []
    for j in range(len(Y)):
        r.append(Y[i].dot(Y[j])*X[i].dot(X[j]))
    P.append(r)
P = matrix(P).T
    
# initialize q
q = []
for i in range(len(Y)):
    q.append(-1.)
q = matrix(q)

# initialize G
G = np.zeros([len(Y),len(Y)])
for i in range(len(Y)):
    G[i][i] = -1.
G0 = copy.copy(G)
G0 = -G0
G = G.tolist()
for i in range(len(Y)):
    G.append(G0[i].tolist())
G = matrix(G).T

# initialize H
h = []
for i in range(len(Y)):
    h.append(0.)
for i in range(len(Y)):
    h.append(C)
h = matrix(h)

# intialize A
A = []
for i in range(len(Y)):
    A.append(Y[i][0])
A = matrix(A).T

# initialize b
b = matrix([0.])
    
# find the solution
solution = solvers.qp(P,q,G,h, A, b)
xvals = np.array(solution['x'])

# remove entries with value < 1e-6, store in new list
xv = []
for i in range(len(Y)):
    if xvals[i] > 1e-6:
        xv.append([xvals[i],Y[i][0],X[i]])

def predictSVMDual(x):
    s = 0
    for e in xv:
        s += e[0]*e[1]*e[2].dot(x)
    return s

# plot training results
plotDecisionBoundary(X, Y, predictSVMDual, [-1, 0, 1], title = 'Dual SVM Train: ' + dataname)

#print '======Validation======'
# load data from csv files
validate = np.loadtxt('data/data_'+ dataname +'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVMDual, [-1, 0, 1], title = 'Dual SVM Validate: ' + dataname)

#####################################################################33
# 3 Kernel SVM

dataname = 'nonlin'
# load data from csv files
train = np.loadtxt('data/data_'+ dataname +'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

C = 1/(2*1e-3)
# creates a P matrix based on a kernel function K
def getP(K):
    P = []
    for i in range(len(Y)):
        r = []
        for j in range(len(Y)):
            r.append(Y[i].dot(Y[j])*K(X[i], X[j]))
        P.append(r)
    P = matrix(P).T
    return P

def gaussian(x1, x2):
    gamma = 5
    return np.exp(-gamma*(np.linalg.norm(x2-x1)**2))

def poly(x1, x2):
    return (1+1e-2*x1.dot(x2))**2

P = getP(poly)

# initialize q
q = []
for i in range(len(Y)):
    q.append(-1.)
q = matrix(q)

# initialize G
G = np.zeros([len(Y),len(Y)])
for i in range(len(Y)):
    G[i][i] = -1.
G0 = copy.copy(G)
G0 = -G0
G = G.tolist()
for i in range(len(Y)):
    G.append(G0[i].tolist())
G = matrix(G).T

# initialize H
h = []
for i in range(len(Y)):
    h.append(0.)
for i in range(len(Y)):
    h.append(C)
h = matrix(h)

# intialize A
A = []
for i in range(len(Y)):
    A.append(Y[i][0])
A = matrix(A).T

# initialize b
b = matrix([0.])

solution = solvers.qp(P,q,G,h, A, b)
xvals = np.array(solution['x'])

# remove entries with value < 1e-6, store in new list
xv = []
for i in range(len(Y)):
    if xvals[i] > 1e-3:
        xv.append([xvals[i],Y[i][0],X[i]])



# Define the predictSVM(x) function, which uses trained parameters
def predictSVMKernel(x):
    s = 0
    for e in xv:
        s += e[0]*e[1]*poly(e[2],x)
    return s




# plot training results
plotDecisionBoundary(X, Y, predictSVMKernel, [-1, 0, 1], title = 'Kernel SVM Train: ' + dataname)

#print '======Validation======'
# load data from csv files
validate = np.loadtxt('data/data_'+ dataname +'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVMKernel, [-1, 0, 1], title = 'Kernel SVM Validate: ' + dataname)