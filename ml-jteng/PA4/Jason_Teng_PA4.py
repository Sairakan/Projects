# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pylab as pl

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

parser = argparse.ArgumentParser()
# -alg ALGORITHM -size SIZE -gamma GAMMA -exps EXPERIMENTS
# -eps EPISODES -epsilon EPSILON -alpha APLHA -lambda LAMBDA
parser.add_argument("-alg", help="Algorithm type")
parser.add_argument("-size", default=5, help="Grid size")
parser.add_argument("-gamma", default=0.99, help="Discount factor")
parser.add_argument("-exps", default=500, help="Number of experiments to run")
parser.add_argument("-eps", default=500, help="Number of learning episodes per experiment")
parser.add_argument("-epsilon", default=0.1, help="Epsilon in e-greedy policy execution")
parser.add_argument("-alpha", default=0.1, help="Learning rate")
parser.add_argument("-lambda", default=0, help="Lambda for SARSA")
parser.add_argument("-plotname", default='', help="File name for plots")

args = parser.parse_args()
alg = args.alg
size = int(args.size)
gamma = float(args.gamma)
exps = int(args.exps)
eps = int(args.eps)
epsilon = float(args.epsilon)
alpha = float(args.alpha)
lam = float(getattr(args, "lambda"))
plotname = args.plotname

class GridworldEnv():
    """
    You are an agent on an s x s grid and your goal is to reach the terminal
    state at the top right corner.
    For example, a 4x4 grid looks as follows:
    o  o  o  T
    o  o  o  o
    o  o  o  o
    x  o  o  o
    
    x is your position and T is the terminal state.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -0.1 at each step until you reach a terminal state.
    """

    def __init__(self, shape=[size,size]):
        self.shape = shape

        nS = np.prod(shape) # The area of the gridworld
        MAX_Y = shape[0]
        MAX_X = shape[1]
        nA = 4  # There are four possible actions
        self.P = {} # P[s][a] = [s1, r(s, a)]; P is the reward table
        grid = np.arange(nS).reshape(shape)
        
        # initialize rewards
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex    # s is the current position id. s = y * 4 + x
            y, x = it.multi_index
            
            self.P[s] = {a : [] for a in range(nA)}
            
            # make states for each possible action, with bounds-checking
            ns_up = s if y == 0 else s - MAX_X
            ns_right = s if x == (MAX_X - 1) else s + 1
            ns_down = s if y == (MAX_Y - 1) else s + MAX_X
            ns_left = s if x == 0 else s - 1
            # update the reward table
            self.P[s][UP] = [ns_up, self.reward(ns_up)]
            self.P[s][RIGHT] = [ns_right, self.reward(ns_right)]
            self.P[s][DOWN] = [ns_down, self.reward(ns_down)]
            self.P[s][LEFT] = [ns_left, self.reward(ns_left)]
                
            it.iternext()
    
    def is_done(self, s):
        return s == size - 1
    
    def reward(self, s):
        return 5.0 if self.is_done(s) else -0.1

    # The possible action has a 0.8 probability of succeeding
    def action_success(self, success_rate = 0.8):
        return np.random.choice(2, 1, p=[1-success_rate, success_rate])[0]
    
    # If the action fails, any action is chosen uniformly(including the succeeding action)
    def get_action(self, action):
        if self.action_success():
            return action
        else:
            random_action = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            return random_action
    
    # Given the current position, this function outputs the position after the action.
    def move(self, s, action):
        return self.P[s][action]

# given a state s, return the policy-decided action (with epsilon chance for a random action)
def choose_action(g, s, Qs):
    if np.random.rand() < epsilon: # random exploration based on epsilon
        return np.random.choice(np.arange(4))
    else:
        qs = [Qs[s][a] for a in range(4)] # get q-values for each action at this state
        maxQ = max(qs)
        if qs.count(maxQ) > 1: # choose randomly if multiple "best" actions based on policy
            best = [a for a in range(4) if qs[a] == maxQ]
            return np.random.choice(best)
        else:
            return qs.index(maxQ)
    
def QLearn(g):
    numSteps = [] # list of steps for each episode
    maxQs = [] # list of the maxQ value of the start state for each episode
    Qs = np.zeros([size*size, 4])
    start = size*(size-1)
    for i in range(eps):
        s = start # initialize s to lower-left corner
        step = 0
        while not g.is_done(s): # not in terminal state
            a = choose_action(g, s, Qs)
            result = g.move(s, g.get_action(a))
            s1 = result[0]
            reward = result[1]
            Qs[s][a] = Qs[s][a] + alpha*(reward + gamma*max([Qs[s1][x] for x in range(4)]) - Qs[s][a])
            s = s1
            step += 1
        numSteps.append(step)
        maxQ = max([Qs[start][x] for x in range(4)])
        maxQs.append(maxQ)
    return numSteps, maxQs

def Sarsa(g):
    numSteps = []
    maxQs = []
    Qs = np.zeros([size*size, 4])
    Es = np.zeros([size*size, 4])
    start = size*(size-1)
    for i in range(eps):
        s = start # initialize s to lower-left corner
        a = choose_action(g, s, Qs)
        step = 0
        while not g.is_done(s):
            result = g.move(s, g.get_action(a))
            s1 = result[0]
            reward = result[1]
            a1 = choose_action(g, s1, Qs)
            delta = reward + gamma*Qs[s1][a1]-Qs[s][a]
            Es[s][a] += 1
            for i in range(size*size):
                for j in range(4):
                    Qs[i][j] = Qs[i][j] + alpha*delta*Es[i][j]
                    Es[i][j] = gamma*lam*Es[i][j]
            s = s1
            a = a1
            step += 1
        numSteps.append(step)
        maxQ = max([Qs[start][x] for x in range(4)])
        maxQs.append(maxQ)
    return numSteps, maxQs

if __name__ == "__main__":
    g = GridworldEnv()
    allSteps = []
    allMaxQs = []
    for i in range(exps):
        if i % 10 == 0:
            print str(i)
        numSteps, maxQs = QLearn(g) if alg == 'q' else Sarsa(g)
        allSteps.append(numSteps)
        allMaxQs.append(maxQs)
    avgSteps = np.mean(allSteps, axis=0)
    avgMaxQs = np.mean(allMaxQs, axis=0)
    pl.figure(1)
    pl.title(plotname+'_steps')
    pl.plot(avgSteps)
    pl.savefig("plots/"+plotname+'_steps.png', bbox_inches='tight')
    pl.figure(2)
    pl.title(plotname+'_maxQ')
    pl.plot(avgMaxQs)
    pl.savefig("plots/"+plotname+'_maxQ.png', bbox_inches='tight')
    
    
    
