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
import os
import numpy as np
import math
import copy
import multiprocessing as mp
import utils
import queue
# import matplotlib.pyplot as plt

class MPPI(object):
    """docstring for MPPI."""
    def __init__(self, args):
        super(MPPI, self).__init__()
        np.random.seed(1)
        self.num_joint = args['JOINTSNUM']
        self.K = args['K']
        self.T = args['T']
        self.alpha = args['alpha']
        self.lamb = args['lamb']
        self.gama = args['gama']
        self.RENDER = args['render']

        #set the env and cost function
        self.set_Cost(args['cost_fun'])
        self.set_Env(args['env_fun'])
        # print(args['mu'])
        if not args['mu'].any():
            self.set_Mu_Sigma(np.zeros(self.num_joint), 0.1*np.eye(self.num_joint))
        else:
            self.set_Mu_Sigma(args['mu'], args['sigma'])

        self.output = mp.Manager().Queue()
        self.low_bound = -1.0
        self.high_bound = 1.0
        self.init_RealEnv(self.RENDER)
        self.init_U()


    def set_Cost(self, cost_fun = None):
        self.CUSTOM_COST = None
        self.CUSTOM_COST = cost_fun

    def set_Mu_Sigma(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def get_Cost(self, data):
        if self.CUSTOM_COST:
            return self.CUSTOM_COST(data)
        else:
            printf("There is no customerized cost function.")
            return sum([x**2] for x in state)

    def get_TerminalCost(self, state):
        return 0

    def get_Normal(self, mu, sigma, T = 1):
        ranVect = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(self.mu, np.diag(self.sigma))]))
        return ranVect

    def set_Env(self, env_fun = None):
        self.CUSTOM_ENV = env_fun

    def get_Env(self):
        if self.CUSTOM_ENV:
            return self.CUSTOM_ENV()
        else:
            printf("There is no customerized cost function.")
            exit()

    def init_RealEnv(self, rend = "None"):
        self.realEnv = self.get_Env()
        self.RENDER = rend
        if self.RENDER == "RENDER":
            self.CUSTOM_VIEWER = MjViewer(self.realEnv)
            self.CUSTOM_VIEWER._render_every_frame = True
            self.CUSTOM_VIEWER._video_idx = 1
        elif self.RENDER == "RECORD":
            self.record=queue.Queue()
        else:
            self.CUSTOM_VIEWER = None

    def record_RealEnv(self):
        self.recordRealEnv = self.realEnv.get_state()

    def init_SimEnv(self, realState):
        simEnv = self.get_Env()
        simEnv.set_state(realState)
        return simEnv

    def update_Control(self):
        for i in range(len(self.U)):
            for j in range(len(self.base_control)):
                self.U[i] += self.base_control[j][i] * self.w[j]
        # return U

    def compute_Weight(self):
        lou = min(self.S)
        yita = sum(math.exp((lou - self.S[x])/self.lamb) for x in range(len(self.S)))
        self.w = []
        for i in range(len(self.S)):
            self.w.append(math.exp((lou - self.S[i])/self.lamb)/yita)
        self.w

    def apply_Control(self, env, ctrl):
        env.data.ctrl[:] = np.clip(ctrl, [self.low_bound]*self.num_joint, [self.high_bound]*self.num_joint)
        env.step()

    def init_U(self):
        self.U = self.get_Normal(self.mu, self.sigma, self.T)

    def add_U(self):
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.get_Normal(self.mu, self.sigma)

    def run_Episode(self, realState, output, k, kexi):
        simEnv = self.init_SimEnv(realState)
        # kexi = self.get_Normal(self.mu, self.sigma, self.T)
        # self.base_control.append(kexi)
        episode_cost = 0

        for t in range(self.T):
            if k < int((1-self.alpha)*self.K):
                v = self.U[t] + kexi[t]
            else:
                v = kexi[t]

            self.apply_Control(simEnv, v)
            episode_cost += self.get_Cost(simEnv.data)#TODO terminal state cost
            # temp.append(sample_sim.data.site_xpos[0])
        episode_cost += self.get_TerminalCost(simEnv.data)#TODO define phi, the terminal cost
        self.output.put((k, episode_cost))

    def run_MPPI(self, iters):
        for i in range(iters):#TODO: implement the taskFinish function
            self.S=[0]*self.K
            self.base_control = []
            processes = []
            self.record_RealEnv()
            # self.init_SimEnv()

            for k in range(self.K):
                kexi = self.get_Normal(self.mu, self.sigma, self.T)
                self.base_control.append(kexi)
                processes.append(mp.Process(target=self.run_Episode, args=(self.recordRealEnv, self.output, k, kexi)))

            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()

            results = [self.output.get() for k in range(self.K)]
            results.sort()
            self.S = [r[1] for r in results]

            # print(self.S)
            self.compute_Weight()
            self.update_Control()
            self.apply_Control(self.realEnv, self.U[0])
            self.add_U()

    # real_sim.data.ctrl[:] = np.clip(U[0], [-0.4]*JOINTSNUM, [0.4]*JOINTSNUM)

    # U[:-1] = U[1:]
    # U[-1] = np.array(np.transpose([np.random.normal(m, s) for m,s in zip(mu, np.diag(sigma))]))

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                self.record.put(np.flip(self.realEnv.render( 1280, 608, device_id = 0), 0))
                
        if self.RENDER == "RECORD":        
            utils.save_video(self.record, ".videos/video_"+utils.getTimeStamp()+".mp4", 10)
            
        print("Finish MPPI Planning")
