import random
import pdb
import math
import pylab as pl
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import numpy as np
import copy
 
#############################
# Mixture Of Gaussians
#############################
 
# A simple class for a Mixture of Gaussians
class MOG:
    def __init__(self, pi = 0, mu = 0, var = 0):
        self.pi = pi
        self.mu = mu
        self.var = var
    def plot(self, color = 'black'):
        return plotGauss2D(self.mu, self.var, color=color)
    def __str__(self):
        return "[pi=%.2f,mu=%s, var=%s]"%(self.pi, self.mu.tolist(), self.var.tolist())
    __repr__ = __str__
 
colors = ('blue', 'yellow', 'black', 'red', 'cyan')

def plotGauss2D(pos, P, r = 2., color = 'black'):
    U, s , Vh = pl.linalg.svd(P)
    orient = math.atan2(U[1,0],U[0,0])*180/math.pi
    ellipsePlot = Ellipse(xy=pos, width=2*r*math.sqrt(s[0]),
              height=2*r*math.sqrt(s[1]), angle=orient,
              edgecolor=color, fill = False, lw = 3, zorder = 10)
    ax = pl.gca()
    ax.add_patch(ellipsePlot);
    return ellipsePlot
 
def plotMOG(X, param, colors = colors, title=""):
    fig = pl.figure()                   # make a new figure/window
    ax = fig.add_subplot(111, aspect='equal')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax.set_xlim(min(x_min, y_min), max(x_max, y_max))
    ax.set_ylim(min(x_min, y_min), max(x_max, y_max))
    ax.set_xlabel("log-likelihood: " + str(loglikelihood(X, param)))
    for (g, c) in zip(param, colors[:len(param)]):
        e = g.plot(color=c)
        ax.add_artist(e)
    plotData(X)
    pl.title(title)
    pl.show()
 
def plotData(X):
    pl.plot(X[:,0:1].T[0],X[:,1:2].T[0], 'gs')
 
def varMat(diag=False):
    A = np.random.normal(loc=0,scale=10,size=(2,2))
    if diag:
        A *= np.identity(2)
    return np.dot(A, A.T)

def dist(a, b):
    return np.linalg.norm(np.subtract(a, b))

"""
Given: indices i and k, a list of data X, and a list of pi, mu, and cov values
Returns: the expectation r[k][i] given the params
"""
def expectation(i, k, X, pis, mus, covs):
    num = pis[k] * multivariate_normal.pdf(X[i], mus[k], covs[k], allow_singular=True)
    denom = 0
    for l in range(len(pis)):
        denom += pis[l] * multivariate_normal.pdf(X[i], mus[l], covs[l], allow_singular=True)
    return num/denom

def esum(k, X, pis, mus, covs):
    ans = 0
    for i in range(len(X)):
        ans += expectation(i, k, X, pis, mus, covs)
    return ans

def nextpi(k, X, pis, mus, covs):
    return esum(k, X, pis, mus, covs) / len(X)

"""
cluster is used for k-means, contains a list of indices corresponding to entries in X
which is in the kth cluster
""" 
def nextmu(k, X, pis, mus, covs,):
    rxsum = 0
    for i in range(len(X)):
        rxsum += expectation(i, k, X, pis, mus, covs) * X[i]
    return rxsum / esum(k, X, pis, mus, covs)

def nextcov(k, X, pis, mus, covs, diag=False):
    rxxsum = 0
    for i in range(len(X)):
        rxxsum += expectation(i, k, X, pis, mus, covs) * np.outer(X[i] - mus[k], (X[i] - mus[k]))
    if diag:
        rxxsum *= np.identity(2)
    return rxxsum / esum(k, X, pis, mus, covs)

def KMeans(X, m=2, threshold = 1e-6):
    (n,d) = X.shape
    mus = [X[random.randint(0,n-1),:] for i in range(m)]
    # clusters[i] == j if X[i] is in cluster j
    clusters = np.zeros(len(X))
    while True:
        # calculate the clusters
        for i in range(len(X)):
            distances = []
            for mu in mus:
                distances.append(dist(X[i], mu))
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # update new means
        nextmus = copy.copy(mus)
        for k in range(m):
            points = [X[i] for i in range(len(X)) if clusters[i] == k]
            nextmus[k] = np.mean(points, axis=0)
        # check if means moved
        if dist(np.array(mus), np.array(nextmus)) < threshold:
            return mus
        else:
            mus = nextmus

"""
Given: a list of data X and (optional) a number of iterations
Returns: a list of MOGs, with each MOG maximized according
    to the EM algorithm.
"""
def EM(X, m=2, threshold = 1e-6, diag=False):
    (n, d) = X.shape
    pis = [1./m for i in range(m)]
    mus = KMeans(X, m)
    covs = [varMat(diag) for i in range(m)]
    MOGs = [MOG(pis[i],mus[i],covs[i]) for i in range(len(pis))]
    ll = loglikelihood(X, MOGs)
    while True:
        nextpis = copy.copy(pis)
        nextmus = copy.copy(mus)
        nextcovs = copy.copy(covs)
        for k in range(m):
            nextpis[k] = nextpi(k, X, pis, mus, covs)
        pis = nextpis 
        for k in range(m):
            nextmus[k] = nextmu(k, X, pis, mus, covs)
        mus = nextmus
        for k in range(m):
            nextcovs[k] = nextcov(k, X, pis, mus, covs, diag)
        covs = nextcovs
        MOGs = [MOG(pis[i],mus[i],covs[i]) for i in range(m)]
        ll2 = loglikelihood(X, MOGs)
        if ll2 - ll < threshold:
            return MOGs
        else:
            ll = ll2

def loglikelihood(X, MOGs):
    pis = [g.pi for g in MOGs]
    mus = [g.mu for g in MOGs]
    covs = [g.var for g in MOGs]
    s = 0
    for i in range(len(X)):
        ps = np.longdouble(0.)
        for l in range(len(pis)):
            ps += pis[l] * multivariate_normal.pdf(X[i], mus[l], covs[l], allow_singular=True)
        s += np.log(ps)
    return s

def avgloglikelihood(X, MOGs):
    pis = [g.pi for g in MOGs]
    mus = [g.mu for g in MOGs]
    covs = [g.var for g in MOGs]
    s = 0
    for i in range(len(X)):
        ps = np.longdouble(0.)
        for l in range(len(pis)):
            ps += pis[l] * multivariate_normal.pdf(X[i], mus[l], covs[l], allow_singular=True)
        s += np.log(ps)
    return s / len(X)

def kcross(dataname, numfolds=5, m=2, diag=False):
    X = np.loadtxt('data/' + dataname + '.txt')
    folds = np.array_split(X, numfolds)
    perf = 0
    for fold in folds:
        training = np.concatenate([x for x in folds if not np.array_equal(x, fold)])
        MOGs = EM(training, m=m, diag=diag)
        perf += loglikelihood(fold, MOGs)
    return perf / numfolds

def makeplot(dataname, diag=False, m=2):
    X = np.loadtxt('data/' + dataname + '.txt')
    MOGs = EM(X, m=m, diag=diag)
    plotMOG(X, MOGs, title=dataname)
    #print "k-cross: " + str(kcross(dataname, m=m))
 
# parameters
dataname = 'data_1_small'
#makeplot(dataname)

dataname = 'data_1_large'
#makeplot(dataname)

dataname = 'data_2_small'
#makeplot(dataname)

dataname = 'data_2_large'
#makeplot(dataname)

dataname = 'data_3_small'
#makeplot(dataname)

dataname = 'data_3_large'
#makeplot(dataname)

dataname = 'mystery_1'
#makeplot(dataname, m=2)
#makeplot(dataname, m=3)
#makeplot(dataname, m=4)
#makeplot(dataname, m=5)

dataname = 'mystery_2'
#makeplot(dataname, m=2)
#makeplot(dataname, m=3)
#makeplot(dataname, m=4)
makeplot(dataname, m=5)

# candidate models
dataname = 'data_1'
X = np.loadtxt('data/' + dataname + '_small.txt')
#MOG_1_diag_1 = EM(X, m=1, diag=True)
#MOG_1_diag_2 = EM(X, m=2, diag=True)
#MOG_1_diag_3 = EM(X, m=3, diag=True)
#MOG_1_diag_4 = EM(X, m=4, diag=True)
#MOG_1_diag_5 = EM(X, m=5, diag=True)
#MOG_1_full_1 = EM(X, m=1, diag=False)
#MOG_1_full_2 = EM(X, m=2, diag=False)
#MOG_1_full_3 = EM(X, m=3, diag=False)
#MOG_1_full_4 = EM(X, m=4, diag=False)
#MOG_1_full_5 = EM(X, m=5, diag=False)
# testing vs large
X = np.loadtxt('data/' + dataname + '_large.txt')
lls = {}
#lls['MOG_1_diag_1'] = avgloglikelihood(X, MOG_1_diag_1)
#lls['MOG_1_diag_2'] = avgloglikelihood(X, MOG_1_diag_2)
#lls['MOG_1_diag_3'] = avgloglikelihood(X, MOG_1_diag_3)
#lls['MOG_1_diag_4'] = avgloglikelihood(X, MOG_1_diag_4)
#lls['MOG_1_diag_5'] = avgloglikelihood(X, MOG_1_diag_5)
#lls['MOG_1_full_1'] = avgloglikelihood(X, MOG_1_full_1)
#lls['MOG_1_full_2'] = avgloglikelihood(X, MOG_1_full_2)
#lls['MOG_1_full_3'] = avgloglikelihood(X, MOG_1_full_3)
#lls['MOG_1_full_4'] = avgloglikelihood(X, MOG_1_full_4)
#lls['MOG_1_full_5'] = avgloglikelihood(X, MOG_1_full_5)
#print lls

dataname = 'data_2'
X = np.loadtxt('data/' + dataname + '_small.txt')
#MOG_2_diag_1 = EM(X, m=1, diag=True)
#MOG_2_diag_2 = EM(X, m=2, diag=True)
#MOG_2_diag_3 = EM(X, m=3, diag=True)
#MOG_2_diag_4 = EM(X, m=4, diag=True)
#MOG_2_diag_5 = EM(X, m=5, diag=True)
#MOG_2_full_1 = EM(X, m=1, diag=False)
#MOG_2_full_2 = EM(X, m=2, diag=False)
#MOG_2_full_3 = EM(X, m=3, diag=False)
#MOG_2_full_4 = EM(X, m=4, diag=False)
#MOG_2_full_5 = EM(X, m=5, diag=False)
# testing vs large
X = np.loadtxt('data/' + dataname + '_large.txt')
lls = {}
#lls['MOG_2_diag_1'] = avgloglikelihood(X, MOG_2_diag_1)
#lls['MOG_2_diag_2'] = avgloglikelihood(X, MOG_2_diag_2)
#lls['MOG_2_diag_3'] = avgloglikelihood(X, MOG_2_diag_3)
#lls['MOG_2_diag_4'] = avgloglikelihood(X, MOG_2_diag_4)
#lls['MOG_2_diag_5'] = avgloglikelihood(X, MOG_2_diag_5)
#lls['MOG_2_full_1'] = avgloglikelihood(X, MOG_2_full_1)
#lls['MOG_2_full_2'] = avgloglikelihood(X, MOG_2_full_2)
#lls['MOG_2_full_3'] = avgloglikelihood(X, MOG_2_full_3)
#lls['MOG_2_full_4'] = avgloglikelihood(X, MOG_2_full_4)
#lls['MOG_2_full_5'] = avgloglikelihood(X, MOG_2_full_5)
#print lls

dataname = 'data_3'
X = np.loadtxt('data/' + dataname + '_small.txt')
#MOG_3_diag_1 = EM(X, m=1, diag=True)
#MOG_3_diag_2 = EM(X, m=2, diag=True)
#MOG_3_diag_3 = EM(X, m=3, diag=True)
#MOG_3_diag_4 = EM(X, m=4, diag=True)
#MOG_3_diag_5 = EM(X, m=5, diag=True)
#MOG_3_full_1 = EM(X, m=1, diag=False)
#MOG_3_full_2 = EM(X, m=2, diag=False)
#MOG_3_full_3 = EM(X, m=3, diag=False)
#MOG_3_full_4 = EM(X, m=4, diag=False)
#MOG_3_full_5 = EM(X, m=5, diag=False)
# testing vs large
X = np.loadtxt('data/' + dataname + '_large.txt')
lls = {}
#lls['MOG_3_diag_1'] = avgloglikelihood(X, MOG_3_diag_1)
#lls['MOG_3_diag_2'] = avgloglikelihood(X, MOG_3_diag_2)
#lls['MOG_3_diag_3'] = avgloglikelihood(X, MOG_3_diag_3)
#lls['MOG_3_diag_4'] = avgloglikelihood(X, MOG_3_diag_4)
#lls['MOG_3_diag_5'] = avgloglikelihood(X, MOG_3_diag_5)
#lls['MOG_3_full_1'] = avgloglikelihood(X, MOG_3_full_1)
#lls['MOG_3_full_2'] = avgloglikelihood(X, MOG_3_full_2)
#lls['MOG_3_full_3'] = avgloglikelihood(X, MOG_3_full_3)
#lls['MOG_3_full_4'] = avgloglikelihood(X, MOG_3_full_4)
#lls['MOG_3_full_5'] = avgloglikelihood(X, MOG_3_full_5)
#print lls

# k-cross-validation
dataname = 'data_1_small'
#klls = {}
#klls['LL_1_diag_1'] = kcross(dataname, m=1, diag=True)
#klls['LL_1_diag_2'] = kcross(dataname, m=2, diag=True)
#klls['LL_1_diag_3'] = kcross(dataname, m=3, diag=True)
#klls['LL_1_diag_4'] = kcross(dataname, m=4, diag=True)
#klls['LL_1_full_1'] = kcross(dataname, m=1, diag=False)
#klls['LL_1_full_2'] = kcross(dataname, m=2, diag=False)
#klls['LL_1_full_3'] = kcross(dataname, m=3, diag=False)
#klls['LL_1_full_4'] = kcross(dataname, m=4, diag=False)
#print klls

dataname = 'data_2_small'
#klls = {}
#klls['LL_2_diag_1'] = kcross(dataname, m=1, diag=True)
#klls['LL_2_diag_2'] = kcross(dataname, m=2, diag=True)
#klls['LL_2_diag_3'] = kcross(dataname, m=3, diag=True)
#klls['LL_2_diag_4'] = kcross(dataname, m=4, diag=True)
#klls['LL_2_full_1'] = kcross(dataname, m=1, diag=False)
#klls['LL_2_full_2'] = kcross(dataname, m=2, diag=False)
#klls['LL_2_full_3'] = kcross(dataname, m=3, diag=False)
#klls['LL_2_full_4'] = kcross(dataname, m=4, diag=False)
#print sorted(klls.items(), key=lambda(k,v):(v,k))

dataname = 'data_3_small'
#klls = {}
#klls['LL_2_diag_1'] = kcross(dataname, m=1, diag=True)
#klls['LL_2_diag_2'] = kcross(dataname, m=2, diag=True)
#klls['LL_2_diag_3'] = kcross(dataname, m=3, diag=True)
#klls['LL_2_diag_4'] = kcross(dataname, m=4, diag=True)
#klls['LL_2_full_1'] = kcross(dataname, m=1, diag=False)
#klls['LL_2_full_2'] = kcross(dataname, m=2, diag=False)
#klls['LL_2_full_3'] = kcross(dataname, m=3, diag=False)
#klls['LL_2_full_4'] = kcross(dataname, m=4, diag=False)
#print sorted(klls.items(), key=lambda(k,v):(v,k))

# leave-1-out
dataname = 'data_1_small'
#klls = {}
#klls['LL_1_diag_1'] = kcross(dataname, m=1, numfolds = 40, diag=True)
#klls['LL_1_diag_2'] = kcross(dataname, m=2, numfolds = 40, diag=True)
#klls['LL_1_diag_3'] = kcross(dataname, m=3, numfolds = 40, diag=True)
#klls['LL_1_diag_4'] = kcross(dataname, m=4, numfolds = 40, diag=True)
#klls['LL_1_full_1'] = kcross(dataname, m=1, numfolds = 40, diag=False)
#klls['LL_1_full_2'] = kcross(dataname, m=2, numfolds = 40, diag=False)
#klls['LL_1_full_3'] = kcross(dataname, m=3, numfolds = 40, diag=False)
#klls['LL_1_full_4'] = kcross(dataname, m=4, numfolds = 40, diag=False)
#print sorted(klls.items(), key=lambda(k,v):(v,k))

dataname = 'data_2_small'
#klls = {}
#klls['LL_2_diag_1'] = kcross(dataname, m=1, numfolds = 40, diag=True)
#klls['LL_2_diag_2'] = kcross(dataname, m=2, numfolds = 40, diag=True)
#klls['LL_2_diag_3'] = kcross(dataname, m=3, numfolds = 40, diag=True)
#klls['LL_2_diag_4'] = kcross(dataname, m=4, numfolds = 40, diag=True)
#klls['LL_2_full_1'] = kcross(dataname, m=1, numfolds = 40, diag=False)
#klls['LL_2_full_2'] = kcross(dataname, m=2, numfolds = 40, diag=False)
#klls['LL_2_full_3'] = kcross(dataname, m=3, numfolds = 40, diag=False)
#klls['LL_2_full_4'] = kcross(dataname, m=4, numfolds = 40, diag=False)
#print sorted(klls.items(), key=lambda(k,v):(v,k))

dataname = 'data_3_small'
#klls = {}
#klls['LL_2_diag_1'] = kcross(dataname, m=1, numfolds = 40, diag=True)
#klls['LL_2_diag_2'] = kcross(dataname, m=2, numfolds = 40, diag=True)
#klls['LL_2_diag_3'] = kcross(dataname, m=3, numfolds = 40, diag=True)
#klls['LL_2_diag_4'] = kcross(dataname, m=4, numfolds = 40, diag=True)
#klls['LL_2_full_1'] = kcross(dataname, m=1, numfolds = 40, diag=False)
#klls['LL_2_full_2'] = kcross(dataname, m=2, numfolds = 40, diag=False)
#klls['LL_2_full_3'] = kcross(dataname, m=3, numfolds = 40, diag=False)
#klls['LL_2_full_4'] = kcross(dataname, m=4, numfolds = 40, diag=False)
#print sorted(klls.items(), key=lambda(k,v):(v,k))






