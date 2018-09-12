#!/usr/bin/env python
import os
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import math
import time
import datetime
import argparse
import copy
#the first 9 -> 9 joints
#the following 6 -> free joint
#no idea for the last qpos

parser = argparse.ArgumentParser()
parser.add_argument("-K", type=int, default=80, help="# expisodes for each time.")
parser.add_argument("-T", type=int, default=50, help="# Time horizon for each episode.")
parser.add_argument("-a", "--alpha",type=float, default=0.1, help="MPPI args.")
parser.add_argument("-l", "--lamb", type=float, default=3, help="MPPI, get w from S.")
parser.add_argument("-g", "--gama", type = float, default=0.5, help="MPPI, arg in actions cost.")
parser.add_argument("-i","--iters", type=int, default=600, help="# Iterations for the code.")
args = parser.parse_args()


JOINTSNUM = 6

K = args.K
T = args.T
alpha = args.alpha#TODO
lamb = args.lamb
gama = args.gama
iters = args.iters
TARGETSTATE = [1, 1 ,1, 0, 0, 0]
TCOST = 10

#TODO add another function that very close not clear.
def cost(state):
    rootxPos = state[0]
    rootzPos = state[1]
    rootyPos = state[2]

    rootxVel = state[9]
    rootzVel = state[10]
    rootyVel = state[11]
    return (rootzPos-0.5)**2


def getTimeStamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def terminalCost(state):
    rootyPos = state[2]
    if abs(rootyPos)>0.5:
        return TCOST
    else:
        return 0
    # return cost

def getNormal(mu, sigma, T = 1):
    temp = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(mu, np.diag(sigma))]))
    return temp

#simulation initial
def simulationInit():
    env = gym.make("HalfCheetah-v2")
    observation = env.reset()
    return env

# the task finish condition
def taskFinish(sim):
    return False

def weightComputation(S):
    lou = min(S)
    yita = sum(math.exp((lou - S[x])/lamb) for x in range(len(S)))
    w = []
    for i in range(len(S)):
        w.append(math.exp((lou - S[i])/lamb)/yita)
    print('lamb:{}'.format(lamb))
    print('low:{}'.format(lou))
    print('yita:{}'.format(yita))
    print(w)
    return w

def updateControl(U, base_control, w):
    for i in range(len(U)):
        for j in range(len(base_control)):
            U[i] += base_control[j][i] * w[j]
    return U

def getFileName():
    return "K:"+str(K)+"-T:"+str(T)+"-iters:"+str(iters)+"-gama:"+str(gama)+"-lamb:"+str(lamb)+"-alpha:"+str(alpha)

real_sim = simulationInit()
#viewer = MjViewer(real_sim)

# mean and standard deviation
mu = np.zeros(JOINTSNUM)
sigma = 0.3*np.eye(JOINTSNUM)

np.random.seed(1)


# kexi = np.random.normal(np.transpose(mu), sigma, T)
#TODO figure out the control input is column vector or not.
#MPPI main function
U = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(mu, np.diag(sigma))]))

for i in range(iters):#TODO: implement the taskFinish function
    S=[0.0]*K
    base_control = []
    for k in range(K):
        sample_sim = copy.deepcopy(real_sim)
        kexi = np.array(np.transpose([np.random.normal(m, s, T) for m,s in zip(mu, np.diag(sigma))]))
        kexi = np.clip(kexi, real_sim.action_space.low, real_sim.action_space.high)
        base_control.append(kexi)
        for t in range(T):
            if k < int((1-alpha)*K):
                v = U[t] + kexi[t]
            else:
                v = kexi[t]
            observation, reward, done, info = sample_sim.step(v)
            # print(cost(observation))
            S[k] += cost(observation)#TODO terminal state cost
            if done:
                S[k] += TCOST#TODO define phi, the terminal cost
                break
        S[k] += terminalCost(observation)

    w = weightComputation(S)
    # plt.plot(range(len(w)), w, 'blue', range(len(w)), S, 'r')
    # plt.show()

    U = updateControl(U, base_control, w)
    ob, re, do, _ = real_sim.step(U[0])
    print("xPos: {}, xVel: {} ".format(ob[0], ob[9]))
    print("zPos: {}, zVel: {} ".format(ob[1], ob[10]))
    real_sim.render()

    U[:-1] = U[1:]
    U[-1] = np.array(np.transpose([np.random.normal(m, s) for m,s in zip(mu, np.diag(sigma))]))

    print("real_sim works well")

print("finish")
