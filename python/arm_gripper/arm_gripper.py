#!/usr/bin/env python
from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
#the first 9 -> 9 joints
#the following 6 -> free joint
#no idea for the last qpos

#Use several method at every time we start the simulation
JOINTSNUM = 9
K = 50
T = 20
alpha = 0.1#TODO
lamb = 0.1
TARGETSTATE = [1, 1 ,1, 0, 0, 0]
gama = 0.5

#TODO add another function that very close not clear.
def cost(state):
    end_pos = state[0]
    obj_pos = state[1]
    target = [0.2, 0.1, 0]
    cost = 0
    for i in range(len(end_pos)):
        cost += (end_pos[i]-obj_pos[i])**2
        cost += (target[i]-obj_pos[i])**2
    # print("end_pos:")
    # print(end_pos)
    # print("obj_pos:")
    # print(obj_pos)
    return cost

def terminalCost(state):
    obj_pos = state[1]
    target = [0.2, 0.1, 0]
    cost = 0
    for o,t in zip(obj_pos, target):
        cost += (o-t)**2
    return cost

def getNormal(mu, sigma, T = 1):
    temp = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(mu, np.diag(sigma))]))
    return temp
    
#simulation initial
def simulationInit(path="/Users/weijiawu/Documents/GitHub/MPPI.jl/python/arm_gripper/arm_claw.xml"):
    model = load_model_from_path(path)
    real_sim = MjSim(model)
    return real_sim

# the task finish condition
def taskFinish(sim):
    return False

def weightComputation(S):
    lou = min(S)
    yita = sum(math.exp((lou - S[x])/lamb) for x in range(len(S)))
    w = []
    for i in range(len(S)):
        w.append(math.exp((lou - S[i])/lamb)/yita)
    return w

def updateControl(U, base_control, w):
    for i in range(len(U)):
        for j in range(len(base_control)):
            U[i] += base_control[j][i] * w[j]
    return U
# def cost(nex
# def cost(next_state):
    # = next_state

real_sim = simulationInit()
#viewer = MjViewer(real_sim)

# mean and standard deviation
mu = np.zeros(JOINTSNUM)
sigma = 5*np.eye(JOINTSNUM)

np.random.seed(1)


# kexi = np.random.normal(np.transpose(mu), sigma, T)
#TODO figure out the control input is column vector or not.
#MPPI main function
U = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(mu, np.diag(sigma))]))
# print(real_sim.get_state())
real_viewer = MjViewer(real_sim)
real_viewer._record_video = True
# real_viewer._video_frames = [60]
for i in range(2):#TODO: implement the taskFinish function
    S=[0]*K
    base_control = []
    sim_state = real_sim.get_state()
    temp = []
    for k in range(K):
        sample_sim = simulationInit()
        sample_sim.set_state(sim_state)
        kexi = np.array(np.transpose([np.random.normal(m, s, T) for m,s in zip(mu, np.diag(sigma))]))
        base_control.append(kexi)
        for t in range(T):
            if k < int((1-alpha)*K):
                v = U[t] + kexi[t]
            else:
                v = kexi[t]
            # print(v)
            sample_sim.data.ctrl[:] = v
            sample_sim.step()
            # print(next_state)
            # print(sample_sim.data.site_xpos)
            S[k] += cost(sample_sim.data.site_xpos)#TODO terminal state cost
            temp.append(sample_sim.data.site_xpos[0])
            # S[k] += cost(next_state) + np.multiply(np.multiply(np.transpose(gama*u[t]), np.linalg.inv(sigma)), v)# TODO both the cost function and the multiply format
        S[k] += terminalCost(sample_sim.data.site_xpos)#TODO define phi, the terminal cost

    w = weightComputation(S)
    # plt.plot(range(len(w)), w, 'blue', range(len(w)), S, 'r')
    # plt.plot(range(len(temp)), [x[0] for x in temp])
    # plt.show()
    U = updateControl(U, base_control, w)
    real_sim.data.ctrl[:] = U[0]
    real_sim.step()

    real_viewer.render()
    U[:-1] = U[1:]
    U[-1] = np.array(np.transpose([np.random.normal(m, s) for m,s in zip(mu, np.diag(sigma))]))

    # print(real_sim.get_state()[1])
    print("real_sim works well")
mujoco_py.mjviewer.save_video(real_viewer._video_frames, "./video_%07d.mp4", 1)
# # set all the 1D list to be the column vector
# while not taskFinish(real_sim):
#     u = [0.01]*JOINTSNUM
#     S = []
#     sim_state = real_sim.get_state()
#     for k in range(K):
#         for t in range(T):
#             if k < int((1-alpha)*K):
#                 v=(u[t] + kexi[t])
#             else:
#                 v=(kexi[t])
#
#             sim.data.ctrl[:] = v
#             next_state = sim.step()
#             S[k] += cost(next_state) + np.multiply(np.multiply(np.transpose(gama*u[t]), np.linalg.inv(sigma)), v)#TODO define cost function
#         S[k] += terminalStateCost(next_state)#TODO define phi, the terminal cost
#     w = computeWeight(S)#TODO define the computeWeight function
#     U = 0
#     for t in range(T):
#         U.append(SGF(sum(np.multiply(w, np.power(kexi, i) for i in range(K)))))
#     real_sim.data.ctrl[:] = U[0]
#     u[:] = U[1:]

# #Lambda calculated from the fig.3 in the paper.
# def computeWeight(S, lambda = 5):
#     lou = min(S)
#     yita = sum(exp(-1/lambda*(S[i] - lou) for i in range(len(S))))
#     w = []
#     for k in range(len(S)):
#         w.append(1/yita*exp(-1/lambda*(S[k]-lou)))
#     return w
#
# #refine this cost function
# def terminalStateCost(state):
#     if np.linalg.norm(state[-7:-1] - TARGETSTATE) < 1:
#         return -100
#     else:
#         return 0
#
#
# def taskFinish(real_sim):
# print(sim_state)
# # print(sim.data.ctrl[:])
# for i in range(1000):
#     sim.data.ctrl[0] = 10
#     sim.step()
# print(sim.get_state())
