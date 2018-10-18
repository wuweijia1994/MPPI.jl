#!/usr/bin/env python
from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import time
import datetime
import multiprocessing as mp

def getTimeStamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#the first 9 -> 9 joints
#the following 6 -> free joint
#no idea for the last qpos

#Use several method at every time we start the simulation
JOINTSNUM = 17
K = 96
T = 100
alpha = 0.1#TODO
lamb = 0.5
gama = 0.5
iters = 650
TCOST=10
PositionOrder = [0, 1, 2]
from queue import Queue
output = mp.Manager().Queue()
np.random.seed(0)

#TODO add another function that very close not clear.
def cost(state):
    pos = state.qpos
    vel = state.qvel

    rootxPos = pos[PositionOrder[0]]
    rootyPos = pos[PositionOrder[1]]
    rootzPos = pos[PositionOrder[2]]

    rootxVel = vel[PositionOrder[0]]

    return  (rootzPos-1.4)**2+0.2*(rootyPos)**2+0.2*(rootxVel-2.0)**2

def terminalCost(state):
    return 0
    # rootyPos = state[2]
    # if abs(rootyPos)>0.3:
    #     return TCOST
    # else:
    #     return 0



#simulation initial

def simulationInit(path="/home/wuweijia/GitHub/MPPI.jl/python/humanoid/humanoid.xml"):

#def simulationInit(path="/Users/weijiawu/Documents/GitHub/MPPI.jl/python/arm_gripper/arm_claw.xml"):
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
def getFileName():
    return "K:"+str(K)+"-T:"+str(T)+"-iters:"+str(iters)+"-gama:"+str(gama)+"-lamb:"+str(lamb)+"-alpha:"+str(alpha)

real_sim = simulationInit()
#viewer = MjViewer(real_sim)

# mean and standard deviation
# JOINTSNUM = 3
mu = np.zeros(JOINTSNUM)
sigma = 0.15*np.eye(JOINTSNUM)

def getNormal(mu, sigma, T = 1):
    temp = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(mu, np.diag(sigma))]))

    return np.clip(temp, [-0.4]*JOINTSNUM, [0.4]*JOINTSNUM)


# kexi = np.random.normal(np.transpose(mu), sigma, T)
#TODO figure out the control input is column vector or not.
#MPPI main function
U = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(mu, np.diag(sigma))]))

real_viewer = MjViewer(real_sim)
real_viewer._record_video = True
real_viewer._render_every_frame = True
real_viewer._video_idx = 1
# real_viewer._show_mocap=False
# real_viewer._video_frames = [60]
def run_episode(sim_state, output, k, kexi):
    sample_sim = simulationInit()
    sample_sim.set_state(sim_state)
    episode_cost = 0

    for t in range(T):
        if k < int((1-alpha)*K):
            v = U[t] + kexi[t]
        else:
            v = kexi[t]
        sample_sim.data.ctrl[:] = v
        sample_sim.step()
        episode_cost += (1+t*1.0/T)*cost(sample_sim.data)#TODO terminal state cost
    # episode_cost += terminalCost(sample_sim.data.qpos)#TODO define phi, the terminal cost

    # l.acquire()
    output.put((k, episode_cost))
    # l.release()
xx, yy, zz = [], [], []
for i in range(iters):#TODO: implement the taskFinish function
    S=[0]*K
    base_control = []
    sim_state = real_sim.get_state()
    temp = []
    # l = mp.Lock()
    processes = []
    for k in range(K):
        kexi = getNormal(mu, sigma, T)
        processes.append(mp.Process(target=run_episode, args=(sim_state, output, k, kexi)))
        base_control.append(kexi)

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    print("finish parallel computing")
    # print(output)
    '''
    while not output.empty():
        print(output.get())
    print("output is empty")
    '''
    results = [output.get() for k in range(K)]
    results.sort()
    S = [r[1] for r in results]

    w = weightComputation(S)
    # print('S: {}'.format(S))
    # print('w: {}'.format(w))
    '''
    plt.subplot(2,1,1)
    plt.plot(range(len(w)), w, 'blue')
    plt.subplot(2,1,2)
    plt.plot(range(len(w)), S, 'r')

    # plt.plot(range(len(temp)), [x[0] for x in temp])
    plt.show()
    '''
    U = updateControl(U, base_control, w)
    real_sim.data.ctrl[:] = np.clip(U[0], [-0.4]*JOINTSNUM, [0.4]*JOINTSNUM)
    real_sim.step()
    # print(U[0])

    U[:-1] = U[1:]
    U[-1] = np.array(np.transpose([np.random.normal(m, s) for m,s in zip(mu, np.diag(sigma))]))
    # xx.append(real_sim.data.qpos[PositionOrder[0]])
    # yy.append(real_sim.data.qpos[PositionOrder[1]])
    # zz.append(real_sim.data.qpos[PositionOrder[2]])


    # print("xPos: {}, yPos: {} ,zPos{}".format(real_sim.data.qpos[PositionOrder[0]], real_sim.data.qpos[PositionOrder[1]], real_sim.data.qpos[PositionOrder[2]]))
    real_viewer.render()

    # print(real_sim.get_state()[1])
    # print("real_sim works well")
# plt.subplot(1,3,1)
# plt.plot(range(len(xx)), xx, 'blue')
# plt.subplot(1,3,2)
# plt.plot(range(len(yy)), yy, 'r')
# plt.subplot(1,3,3)
# plt.plot(range(len(zz)), zz, 'black')
# # plt.plot(range(len(temp)), [x[0] for x in temp])
# plt.show()

mujoco_py.mjviewer.save_video(real_viewer._video_queue, "./video_"+getFileName()+getTimeStamp()+".mp4", 10)
print("finish")
