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
import queue

def getTimeStamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#the first 9 -> 9 joints
#the following 6 -> free joint
#no idea for the last qpos

#Use several method at every time we start the simulation
JOINTSNUM = 6
K = 50
T = 50
alpha = 0.1#TODO
lamb = 0.1
TARGETSTATE = [1, 1 ,1, 0, 0, 0]
gama = 0.5
iters = 250
TCOST=10
#TODO add another function that very close not clear.
def cost(state):
    rootxPos = state[0]
    rootzPos = state[1]
    rootyPos = state[2]

    # rootxVel = state[9]
    # rootzVel = state[10]
    # rootyVel = state[11]
    return (rootxPos-5)**2

def terminalCost(state):
    rootyPos = state[2]
    if abs(rootyPos)>0.3:
        return TCOST
    else:
        return 0

def getNormal(mu, sigma, T = 1):
    temp = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(mu, np.diag(sigma))]))
    return temp

#simulation initial

def simulationInit(path="/home/wuweijia/GitHub/MPPI/python/half_cheetah/half_cheetah.xml"):

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
mu = np.zeros(JOINTSNUM)
sigma = 1*np.eye(JOINTSNUM)

np.random.seed(1)


# kexi = np.random.normal(np.transpose(mu), sigma, T)
#TODO figure out the control input is column vector or not.
#MPPI main function
U = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(mu, np.diag(sigma))]))
# print(real_sim.get_state())
# pool = MjRenderPool(real_sim, n_workers=4)

#real_viewer = MjViewer(real_sim)
#real_viewer._record_video = True
#real_viewer._render_every_frame = True
#real_viewer._video_idx = 1
# real_viewer._show_mocap=False
# real_viewer._video_frames = [60]
record = queue.Queue()
for i in range(iters):#TODO: implement the taskFinish function
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
            sample_sim.data.ctrl[:] = v
            sample_sim.step()
            S[k] += cost(sample_sim.data.qpos)#TODO terminal state cost
            # temp.append(sample_sim.data.site_xpos[0])
        S[k] += terminalCost(sample_sim.data.qpos)#TODO define phi, the terminal cost

    w = weightComputation(S)
    # plt.plot(range(len(w)), w, 'blue', range(len(w)), S, 'r')
    # plt.plot(range(len(temp)), [x[0] for x in temp])
    # plt.show()
    U = updateControl(U, base_control, w)
    real_sim.data.ctrl[:] = U[0]
    real_sim.step()

    U[:-1] = U[1:]
    U[-1] = np.array(np.transpose([np.random.normal(m, s) for m,s in zip(mu, np.diag(sigma))]))
    #real_viewer.render()
    record.put(np.flip(real_sim.render(1024, 512, device_id = 0), 0))

    # print(real_sim.get_state()[1])
    print("real_sim works well")

import os
import imageio
filename = "./half_cheetah_video.mp4"
if not os.path.isdir(os.path.dirname(filename)):
    os.mkdir(os.path.dirname(filename))

writer = imageio.get_writer(filename, fps=10)
while not record.empty():
    frame = record.get()
    writer.append_data(frame)
writer.close()

#mujoco_py.mjviewer.save_video(real_viewer._video_queue, "./video_"+getFileName()+getTimeStamp()+".mp4", 10)
print("finish")
