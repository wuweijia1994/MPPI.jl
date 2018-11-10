# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.1
# ---

from mujoco_py import load_model_from_path, MjSim, MjViewer
import MPPI
import os
import numpy as np
#from keras.models import Sequential, Model
#from keras.layers import Dense, Activation, Flatten, Input, Concatenate
#from keras.optimizers import Adam

import argparse

parser = argparse.ArgumentParser(description='Process which environment to simulate.')
parser.add_argument('-e', '--env', type=str, nargs='?', default="humanoid",
                    help='Enter the name of the environments like: inverted_pendulum, humanoid')
parser.add_argument('-it', '--iter', type=int, default=500,
                    help='The number of the iterations')

args = parser.parse_args()
ENV = args.env
ITER = args.iter

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

def hm_cost(data):
    pos = data.qpos
    vel = data.qvel

    rootxPos = pos[0]
    rootyPos = pos[1]
    rootzPos = pos[2]

    rootxVel = vel[0]

    return  (rootzPos-1.4)**2+0.2*(rootyPos)**2+0.2*(rootxVel-2.0)**2

def hm_env(path=os.path.join(os.curdir, "humanoid/humanoid.xml")):
    model = load_model_from_path(path)
    real_sim = MjSim(model)
    return real_sim

ip_args = {"JOINTSNUM":1, "K":20, "T":500, "alpha":0.1, "lamb":0.05, "gama":0.5, "render":"Record", "cost_fun":ip_cost, "env_fun":ip_env, "mu":None, "sigma":None}

hm_args = {"JOINTSNUM":17, "K":96, "T":100, "alpha":0.1, "lamb":0.05, "gama":0.5, "render":"RECORD", "cost_fun":hm_cost, "env_fun":hm_env, "mu":np.zeros(17), "sigma":0.2*np.eye(17)}


args = {"inverted_pendulum":ip_args, "humanoid":hm_args}

if ENV not in args:
    print("There is no environment: "+ENV)
else:
    env_arg = args[ENV]

mppi_agent = MPPI.MPPI(env_arg)
mppi_agent.run_MPPI(ITER)
