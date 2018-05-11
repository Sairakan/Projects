# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:58:41 2018

@author: Jason Teng
"""

import numpy as np
import scipy

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

print(scipy.optimize.fmin_bfgs(f3,[0],))



