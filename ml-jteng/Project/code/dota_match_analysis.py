# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:55:22 2018

@author: Jason Teng
"""

import pandas
import numpy as np
import itertools
import pylab as pl
import time

# constants
lr = 0.01
hidden_layer_size = 20

# seeding rng for reproducability
np.random.seed(1)

# returns either a sigmoid or its derivative
def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

# converts a list of hero ids into an input vector for the NN
def reformat(L):
    a = np.zeros(121)
    for x in L:
        a[x] = 1
    return a


# full batch training
def batch_train((X, Y), epochs=10000, lr=0.01, output_size = 1):
    # initialize weights
    # syn0 is the weight matrix from the input layer to the first hidden layer
    syn0 = 2*np.random.random((121,hidden_layer_size)) - 1
    # b0 is the bias vector for the first hidden layer
    b0 = 2*np.random.random((1,hidden_layer_size)) - 1
    # syn1 is the weight matrix from the first hidden layer to the second hidden layer
    syn1 = 2*np.random.random((hidden_layer_size,hidden_layer_size)) - 1
    # b1 is the bias vector for the second hidden layer
    b1 = 2*np.random.random((1,hidden_layer_size)) - 1
    # syn2 is the weight matrix from the second hidden layer to the output layer
    syn2 = 2*np.random.random((hidden_layer_size,output_size)) - 1
    # b2 is the bias vector for the output layer
    b2 = 2*np.random.random((1,output_size)) - 1
    errors = []
    for j in range(epochs):
        
        # forward propagation
        l0 = X
        l1 = sigmoid(np.dot(l0,syn0)+b0)
        l2 = sigmoid(np.dot(l1,syn1)+b1)
        l3 = sigmoid(np.dot(l2,syn2)+b2)
        
        # calculate error
        l3_error = Y - l3
        errors.append(np.mean(np.abs(l3_error)))
        if j % (epochs/10) == 0:
            print "Training Error (epoch " + str(j) + "): " + str(np.mean(np.abs(l3_error)))
        # backpropagation
        l3_delta = l3_error*sigmoid(l3,deriv=True)
        l2_error = l3_delta.dot(syn2.T)
        l2_delta = l2_error*sigmoid(l2,deriv=True)
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error*sigmoid(l1,deriv=True)
        # update weights and biases
        syn2 += l2.T.dot(l3_delta)*lr
        b2 += np.sum(l3_delta,axis=0,keepdims=True)*lr
        syn1 += l1.T.dot(l2_delta)*lr
        b1 += np.sum(l2_delta,axis=0,keepdims=True)*lr
        syn0 += l0.T.dot(l1_delta)*lr
        b0 += np.sum(l1_delta,axis=0,keepdims=True)*lr
    pl.figure()
    pl.title("Batch Training Error vs Epoch #")
    pl.plot(errors)
    pl.show()
    return syn0, b0, syn1, b1, syn2, b2

# online training
def online_train((X, Y), epochs=1000, lr = 1):
    # initialize weights
    # syn0 is the weight matrix from the input layer to the first hidden layer
    syn0 = 2*np.random.random((121,hidden_layer_size)) - 1
    # b0 is the bias vector for the first hidden layer
    b0 = 2*np.random.random((1,hidden_layer_size)) - 1
    # syn1 is the weight matrix from the first hidden layer to the second hidden layer
    syn1 = 2*np.random.random((hidden_layer_size,hidden_layer_size)) - 1
    # b1 is the bias vector for the second hidden layer
    b1 = 2*np.random.random((1,hidden_layer_size)) - 1
    # syn2 is the weight matrix from the second hidden layer to the output layer
    syn2 = 2*np.random.random((hidden_layer_size,1)) - 1
    # b2 is the bias vector for the output layer
    b2 = 2*np.random.random((1,1)) - 1
    errors = []
    for j in range(epochs):
        l0 = X
        l1 = sigmoid(np.dot(l0,syn0)+b0)
        l2 = sigmoid(np.dot(l1,syn1)+b1)
        l3 = sigmoid(np.dot(l2,syn2)+b2)
        l3_error = Y - l3
        errors.append(np.mean(np.abs(l3_error)))
        if j % (epochs/10) == 0:
            print "Training Error (epoch " + str(j) + "): " + str(np.mean(np.abs(l3_error)))
        for x,y in itertools.izip(X,Y):
            
            # forward propagation
            l0 = x
            l1 = sigmoid(np.dot(l0,syn0)+b0)
            l2 = sigmoid(np.dot(l1,syn1)+b1)
            l3 = sigmoid(np.dot(l2,syn2)+b2)
            
            # backpropagation
            l3_error = y - l3
            l3_delta = l3_error*sigmoid(l3,deriv=True)
            l2_error = l3_delta.dot(syn2.T)
            l2_delta = l2_error*sigmoid(l2,deriv=True)
            l1_error = l2_delta.dot(syn1.T)
            l1_delta = l1_error*sigmoid(l1,deriv=True)
            # update weights and biases
            syn2 += l2.T.dot(l3_delta)*lr
            b2 += np.sum(l3_delta,axis=0,keepdims=True)*lr
            syn1 += l1.T.dot(l2_delta)*lr
            b1 += np.sum(l2_delta,axis=0,keepdims=True)*lr
            syn0 += np.outer(l0,l1_delta)*lr
            b0 += np.sum(l1_delta,axis=0,keepdims=True)*lr
    
    pl.figure()
    pl.title("Online Training Error vs Epoch #")
    pl.plot(errors)
    pl.show()
    return syn0, b0, syn1, b1, syn2, b2

# get data
data = pandas.read_csv("dota_match_data.csv")
# only interested in hero and win
del data['match_id']
del data['start_time']
del data['account_id']
del data['leaguename']
# each sample contains the data for a single team in a single game
samples = [data[n:n+5] for n in range(0, len(data), 5)]

# generates training data for win/loss
def parse_samples_win(samples):
    # inputs
    X = np.array([reformat(df['hero_id']) for df in samples])
    # outputs - 1 if win is True, 0 if false
    Y = np.array([[1] if df['win'].iloc[0] else [0] for df in samples])
    return X, Y

# generates training data for missing hero
def parse_samples_last_hero(samples):
    # only interested in wins
    wins = [s['hero_id'] for s in samples if s['win'].any()]
    # inputs
    # for each df in wins, get all combinations of 4 heroes
    X = []
    for df in wins:
        combs = list(itertools.combinations(df, 4))
        for comb in combs:
            X.append(reformat(comb))
    X = np.array(X)
    # output is missing hero; combinations are generated in-order, so missing
    # hero is in reverse order for each original sample
    Y = []
    for df in wins:
        for hero_id in df[::-1]:
            Y.append(reformat([hero_id]))
    Y = np.array(Y)
    return X, Y

# get test data
test_data = pandas.read_csv("dota_match_test.csv")
# only interested in hero and win
del test_data['match_id']
del test_data['start_time']
del test_data['account_id']
del test_data['leaguename']
# each sample contains the data for a single team in a single game
test_samples = [test_data[n:n+5] for n in range(0, len(test_data), 5)]

# takes a list of NN weights and biases, and a parser to read the test samples
def test((syn0, b0, syn1, b1, syn2, b2), parser):
    # use samples to generate training data
    # test inputs
    X_test , Y_test = parser(test_samples)
    
    l0 = X_test
    l1 = sigmoid(np.dot(l0,syn0)+b0)
    l2 = sigmoid(np.dot(l1,syn1)+b1)
    l3 = sigmoid(np.dot(l2,syn2)+b2)
    
    # calculate error
    l3_error = Y_test - l3
    
    print "Test Error:" + str(np.mean(np.abs(l3_error)))

startTime = time.time()
#test(batch_train(parse_samples_win(samples)), parse_samples_win)
print time.time() - startTime
#test(batch_train(parse_samples_last_hero(samples), epochs=10000, lr=0.1, output_size=121), parse_samples_last_hero)

startTime = time.time()
#test(online_train(parse_samples_win(samples)), parse_samples_win)
print time.time() - startTime


test(online_train(parse_samples_win(samples), lr=10), parse_samples_win)





