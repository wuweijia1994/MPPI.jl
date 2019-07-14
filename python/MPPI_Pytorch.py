from mujoco_py import load_model_from_path, load_model_from_mjb, MjSim, MjViewer
import io
import os
import numpy as np
import math
import copy
import multiprocessing as mp
import utils
import queue
import time
import threading
import Net_Model as nm
import scipy as sp

np.random.seed(int(time.time()))
# import matplotlib.pyplot as plt

def get_Cost(name, data):
    if name == "half_cheetah":
        state = data.qpos
        rootxPos = state[0]
        rootzPos = state[1]
        rootyPos = state[2]

        return (rootxPos-5)**2 + 0.3*(rootyPos)**2
    elif name == "robot_arm":
        state= data.site_xpos
        end_pos = state[0]
        obj_pos = state[1]
        target = [0.2, 0.1, 0.2]
        cost = 0
        for i in range(len(end_pos)):
            cost += (end_pos[i]-obj_pos[i])**2
            cost += (target[i]-obj_pos[i])**2
        return cost
    elif name == "humanoid":
        PositionOrder = [0, 1, 2]
        pos = data.qpos
        vel = data.qvel

        rootxPos = pos[PositionOrder[0]]
        rootyPos = pos[PositionOrder[1]]
        rootzPos = pos[PositionOrder[2]]

        rootxVel = vel[PositionOrder[0]]

        return  (rootzPos-1.4)**2+0.2*(rootyPos)**2+0.2*(rootxVel-2.0)**2

    else:
        print("no env")
        exit(0)

def get_TerminalCost(name, data):
    if name == "half_cheetah":
        state = data.qpos
        rootyPos = state[2]
        if abs(rootyPos)>0.3:
            return 100
        else:
            return 0

    elif name == "robot_arm":
        state = data.site_xpos
        obj_pos = state[1]
        target = [0.1, 0.1, 0]
        cost = 0
        for o,t in zip(obj_pos, target):
            cost += (o-t)**2
        return cost
    elif name == "humanoid":
        return 0

    else:
        print("no env")
        exit(0)


def run_Episode(T, alpha, K, U, simState, env_path, output, k, kexi, name):
    model = load_model_from_path(env_path)
    simEnv = MjSim(model)
    simEnv.set_state(simState)

    episode_cost = 0

    for t in range(T):
        if k < int((1-alpha)*K):
            v = U[t] + kexi[t]
        else:
            v = kexi[t]

        simEnv.data.ctrl[:] = v
        simEnv.step()
        episode_cost += get_Cost(name, simEnv.data)#TODO terminal state cost
    episode_cost += get_TerminalCost(name, simEnv.data)#TODO define phi, the terminal cost
    output.put((k, episode_cost))

class MPPI(object):
    """docstring for MPPI."""
    def __init__(self, args):
        super(MPPI, self).__init__()
        #np.random.seed(1)
        self.num_joints = args['JOINTSNUM']
        self.K = args['K']
        self.T = args['T']
        self.alpha = args['alpha']
        self.lamb = args['lamb']
        self.gama = args['gama']
        self.RENDER = args['render']
        self.env_path = args['env_path']
        if "half_cheetah" in self.env_path:
            self.envName = "half_cheetah"
        elif "arm" in self.env_path:
            self.envName = "robot_arm"
        elif "humanoid" in self.env_path:
            self.envName = "humanoid"
        else:
            print("no env")
            exit(0)

        #set the env and cost function
        if not args['mu'].any():
            self.set_Mu_Sigma(np.zeros(self.num_joints), 5*np.eye(self.num_joints))
        else:
            self.set_Mu_Sigma(args['mu'], args['sigma'])

#         self.output = mp.Manager().Queue()
        self.output = mp.Queue()
        self.low_bound = -1.0
        self.high_bound = 1.0
        self.init_RealEnv(self.RENDER)
        self.init_U()
        #import pdb; pdb.set_trace()
        self.num_states = len(self.realEnv.data.qpos) + len(self.realEnv.data.qvel)
        self.create_memory()

    def set_Mu_Sigma(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def create_memory(self):
        self.batch_states = []
        self.batch_actions = []
        self.net = nm.Net(self.num_joints, self.num_states)

    def train_memory(self, x, y):#Set the auto-detect of the batch op
        self.net.train(x, y)

    def predict_memory(self, x):
        return self.net.predict(x)

    def predict_trajectory(self, x):
        _traj = []
        _model = load_model_from_path(self.env_path)
        _sim = MjSim(_model)
        _sim.set_state(x)

        for _i in range(self.T):
            _v = self.predict_memory(_sim.get_state().flatten())
            _traj.append(v)
            _sim.data.ctrl[:] = _v
            _sim.step()

        return _traj

    def record_trajectory(self, info, value):
        if info == "state":
            self.batch_states.append(value)
        elif info == "action":
            self.batch_actions.append(value)
        else:
            print("Unrecognized info to record!")

    def get_Normal(self, mu, sigma, T = 1):
        ranVect = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(self.mu, np.diag(self.sigma))]))
        return ranVect

    def get_fast_normal(self, T):
        disturb = np.random.normal(0, 5, (T, self.num_joints))
        return disturb

    def get_Env(self):
        if self.env_path:
           model = load_model_from_path(self.env_path)
           real_sim = MjSim(model)
           return real_sim
        else:
            print("There is no customerized cost function.")
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

    def update_Control(self, U, base_control, w):
        for i in range(len(U)):
            for j in range(len(base_control)):
                U[i] += base_control[j][i] * w[j]
        return U

    def compute_Weight(self, S):
        lou = min(S)
        yita = sum(math.exp((lou - S[x])/self.lamb) for x in range(len(S)))
        w = []
        w_append = w.append
        for i in range(len(S)):
            w_append(math.exp((lou - S[i])/self.lamb)/yita)
        return w
#         return self.w

    def apply_Control(self, env, ctrl):
        env.data.ctrl[:] = ctrl
        #env.data.ctrl[:] = np.clip(ctrl, [self.low_bound]*self.num_joints, [self.high_bound]*self.num_joints)
        env.step()

    def init_U(self):
        self.U = self.get_Normal(self.mu, self.sigma, self.T)

    def add_U(self, U):
        U[:-1] = U[1:]
        U[-1] = self.get_Normal(self.mu, self.sigma)
        return U

    def run_MPPI(self, iters):
        totalCost = []
        #import time;t1 = time.time(); iters=1; self.K=500
        for i in range(iters):#TODO: implement the taskFinish function
            self.S=[0]*self.K
            self.base_control = []
            processes = []
            self.record_RealEnv()
            # self.init_SimEnv()
            totalCost.append(get_Cost(self.envName, self.realEnv.data))

            for k in range(self.K):
                kexi = self.get_Normal(self.mu, self.sigma, self.T)
                self.base_control.append(kexi)
                _t_simState = self.realEnv.get_state()
                processes.append(mp.Process(target=run_Episode, args=(self.T, self.alpha, self.K, self.U, _t_simState, self.env_path, self.output, k, kexi, self.envName)))

                #processes.append(threading.Thread(target=run_Episode, args=(self.T, self.alpha, self.K, self.U, _t_simEnv, self.output, k, kexi, self.envName)))

            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()

            results = [self.output.get() for k in range(self.K)]
            results.sort()
            self.S = [r[1] for r in results]

            print(self.S[0])
            self.compute_Weight()
            self.update_Control()
            #import pdb; pdb.set_trace()
            self.apply_Control(self.realEnv, self.U[0])
#             print(U[0])
            self.add_U()

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                self.record.put(np.flip(self.realEnv.render(1024, 512, device_id = 0), 0))

        #print(time.time()-t1)
        if self.RENDER == "RECORD":
            _fileName = utils.getTimeStamp()
            utils.save_video(self.record, "./videos/video_"+_fileName+".mp4", 10)
            np.savetxt("./data/Real_"+_fileName+".txt", np.asarray(totalCost))

#         print("Finish MPPI Planning")

    def run_MPPI_get_trajectory(self, iters, init_state):
        realEnv = self.init_RealEnv()
        realEnv.data.qpos[:] = init_state["qpos"]; realEnv.data.qvel[:] = init_state["qvel"]
        realEnv.forward()

        totalCost = []
        control_trajectory = {"states":[], "actions":[]}
        base_control = []
        _output = mp.Queue()
        U = self.get_Normal(self.mu, self.sigma, self.T)

        for _i in range(iters):
            processes = []
            totalCost.append(get_Cost(self.envName, realEnv.data))

            for k in range(self.K):
                kexi = self.get_Normal(self.mu, self.sigma, self.T)
                base_control.append(kexi)
                _t_simState = realEnv.get_state()
                processes.append(mp.Process(target=run_Episode, args=(self.T, self.alpha, self.K, self.U, _t_simState, self.env_path, _output, k, kexi, self.envName)))

            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()

            results = [_output.get() for k in range(self.K)]
            results.sort()
            S = [r[1] for r in results]

            print(S[0])
            W = self.compute_Weight()
            U = self.update_Control(U, base_control, W)
            control_trajectory["states"].append(realEnv.get_state().flatten())#TODO
            control_trajectory["actions"].append(U[0])
            self.apply_Control(realEnv, U[0])
            U = self.add_U()

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                self.record.put(np.flip(self.realEnv.render(1024, 512, device_id = 0), 0))

        return control_trajectory

    def run_GPS(self, iters):
        init_states = 0
        while True:
            _net_states = self.net.predict(init_states)
            _traj = self.run_MPPI_get_trajectory(_iters, _net_states)
            self.net.train(_traj["state"], _traj["action"])
            _last_net_states  = _net_states
            _net_states_kl_divergence = scipy.stats.entropy(_last_net_states, _net_states)

            if _net_states_kl_divergence < 1.0:
                break
        print("Finish the task")

if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import sys
    parser = argparse.ArgumentParser(description='Process which environment to simulate.')
    parser.add_argument('-e', '--env', type=str, nargs='?', default="arm_gripper",
                        help='Enter the name of the environments like: inverted_pendulum, humanoid')
    parser.add_argument('-it', '--iter', type=int, default=300,
                        help='The number of the iterations')

    args = parser.parse_args()
    ENV = args.env
    ITER = args.iter

    hm_env_path = "/home/wuweijia/GitHub/MPPI/python/humanoid/humanoid.xml"
    hc_env_path = "/home/wuweijia/GitHub/MPPI/python/half_cheetah/half_cheetah.xml"
    ag_env_path = "/home/wuweijia/GitHub/MPPI/python/arm_gripper/arm_claw.xml"

    #ip_args = {"JOINTSNUM":1, "K":20, "T":500, "alpha":0.1, "lamb":0.05, "gama":0.5, "render":"RECORD", "env_path":ip_env, "mu":None, "sigma":None}

    hm_args = {"JOINTSNUM":17, "K":200, "T":100, "alpha":0.1, "lamb":0.5, "gama":0.5, "render":"RECORD", "env_path":hm_env_path, "mu":np.zeros(17), "sigma":0.05*np.eye(17)}

    ag_args = {"JOINTSNUM":9, "K":50, "T":50, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":ag_env_path, "mu":np.zeros(9), "sigma":5*np.eye(9)}

    hc_args = {"JOINTSNUM":6, "K":50, "T":50, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":hc_env_path, "mu":np.zeros(6), "sigma":5*np.eye(6)}

    args = {"humanoid":hm_args, "robot_arm":ag_args, "half_cheetah":hc_args}

    if ENV not in args:
        print("There is no environment: "+ENV)
    else:
        env_arg = args[ENV]

    mppi_agent = MPPI(env_arg)
    mppi_agent.run_MPPI(ITER)

