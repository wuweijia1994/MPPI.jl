from mujoco_py import load_model_from_path, MjSim, MjViewer
import MPPI_LR
#import MPPI
#import MPPI_MMR
import os
import numpy as np
#from keras.models import Sequential, Model
#from keras.layers import Dense, Activation, Flatten, Input, Concatenate
#from keras.optimizers import Adam

import argparse
import sys
sys.path.append('../ModelSampling')

# +
parser = argparse.ArgumentParser(description='Process which environment to simulate.')
parser.add_argument('-e', '--env', type=str, nargs='?', default="arm_gripper",
                    help='Enter the name of the environments like: inverted_pendulum, humanoid')
parser.add_argument('-it', '--iter', type=int, default=300,
                    help='The number of the iterations')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
args = parser.parse_args()
ENV = args.env
ITER = args.iter

# +
def ip_cost(data):
    #pole init pos is 0, vel is 0
    state = data.qpos
    cart_pos, pole_pos = state[0], state[1]
    # cart_vel, pole_vel = state[2]
    cost = (pole_pos)*(pole_pos)+(cart_pos)*(cart_pos)
    return cost

def ip_env(path="./inverted_pendulum/inverted_pendulum.xml"):
    model = load_model_from_path(path)
    real_sim = MjSim(model)
    return real_sim

# +
def hm_cost(data):
    pos = data.qpos
    vel = data.qvel

    rootxPos = pos[0]
    rootyPos = pos[1]
    rootzPos = pos[2]

    rootxVel = vel[0]

    return  (rootzPos-1.4)**2
    #return  (rootzPos-1.4)**2+0.01*(rootyPos)**2+0.01*(rootxVel-2.0)**2

def hm_env(path=os.path.join(os.curdir, "humanoid/humanoid.xml")):
    model = load_model_from_path(path)
    real_sim = MjSim(model)
    return real_sim
# hm_env_mjb = path=os.path.join(os.curdir, "humanoid/humanoid.mjb")
def hc_env(path=os.path.join(os.curdir, "half_cheetah/half_cheetah.xml")):
    model = load_model_from_path(path)
    real_sim = MjSim(model)
    return real_sim

hm_env_path = "/home/wuweijia/GitHub/MPPI/python/humanoid/humanoid.xml"
hc_env_path = "/home/wuweijia/GitHub/MPPI/python/half_cheetah/half_cheetah.xml"
ag_env_path = "/home/wuweijia/GitHub/MPPI/python/arm_gripper/arm_claw.xml"

# +
def ag_cost(data):
    state = data.site_xpos

    end_pos = state[0]
    obj_pos = state[1]
    target = [0.2, 0.1, 0.2]

    for i in range(len(end_pos)):
        episode_cost += (end_pos[i]-obj_pos[i])**2
        episode_cost += (target[i]-obj_pos[i])**2

def hc_cost(data):
    state_x = data.qpos[0]

    return (state_x-5)**2

# +
ip_args = {"JOINTSNUM":1, "K":50, "T":500, "alpha":0.1, "lamb":0.05, "gama":0.5, "render":"RECORD", "cost_fun":ip_cost, "env_path":ip_env, "mu":None, "sigma":None}

hm_args = {"JOINTSNUM":17, "K":196, "T":100, "alpha":0.1, "lamb":0.05, "gama":0.5, "render":"RECORD", "cost_fun":hm_cost, "env_path":hm_env_path, "mu":np.zeros(17), "sigma":0.05*np.eye(17)}

ag_args = {"JOINTSNUM":9, "K":50, "T":150, "alpha":0.1, "lamb":5, "gama":0.5, "render":"RECORD", "cost_fun":hm_cost, "env_path":ag_env_path, "mu":np.zeros(9), "sigma":5*np.eye(9)}

hc_args = {"JOINTSNUM":6, "K":50, "T":50, "alpha":0.1, "lamb":5, "gama":0.5, "render":"RECORD", "cost_fun":hc_cost, "env_path":hc_env_path, "mu":np.zeros(6), "sigma":5*np.eye(6)}

args = {"inverted_pendulum":ip_args, "humanoid":hm_args, "arm_gripper":ag_args, "half_cheetah":hc_args}
# -

if ENV not in args:
    print("There is no environment: "+ENV)
else:
    env_arg = args[ENV]
mppi_agent = MPPI_LR.MPPI(env_arg)
mppi_agent.run_MPPI(ITER)

#mppi_agent = MPPI_MMR.MPPI_MMR(env_arg)
## mppi_agent = MPPI.MPPI(env_arg)
## mppi_agent.run_MPPI(ITER)
## mppi_agent.run_MPPI_Supervising(ITER)
#mppi_agent.run_MPPI_GPS_dual(ITER)
