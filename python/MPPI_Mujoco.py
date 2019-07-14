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
import datetime

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

    elif name == "arm_door":
        end_above, end, handle_above, handle = data.site_xpos
        cost = 0

        for i in range(len(end_above)):
            cost += (end_above[i]-handle_above[i])**2
            cost += (end[i]-handle[i])**2
        cost += (handle[1] -0.3)**2
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

    elif name == "arm_door":
        end_above, end, handle_above, handle = data.site_xpos
        cost = 0
        for i in range(len(end_above)):
            cost += (end_above[i]-end[i])**2
            cost += (handle_above[i]-handle[i])**2
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
        self.num_joint = args['JOINTSNUM']
        self.K = args['K']
        self.T = args['T']
        self.alpha = args['alpha']
        self.lamb = args['lamb']
        self.gama = args['gama']
        self.RENDER = args['render']
        self.env_path = args['env_path']
        if "half_cheetah" in self.env_path:
            self.envName = "half_cheetah"
        elif "arm_claw" in self.env_path:
            self.envName = "robot_arm"
        elif "door" in self.env_path:
            self.envName = "arm_door"
        elif "humanoid" in self.env_path:
            self.envName = "humanoid"
        else:
            print("no env")
            exit(0)

        #set the env and cost function
        # print(args['mu'])
        if not args['mu'].any():
            self.set_Mu_Sigma(np.zeros(self.num_joint), 5*np.eye(self.num_joint))
        else:
            self.set_Mu_Sigma(args['mu'], args['sigma'])

#         self.output = mp.Manager().Queue()
        self.output = mp.Queue()
        self.low_bound = -1.0
        self.high_bound = 1.0
        self.init_RealEnv(self.RENDER)
        self.init_U()
        if "states" in args:
            self.set_states(args["states"])
        self.data = {"states":[], "actions":[]}
        if "log" in args:
            self.log = args["log"]
            self.start_time = datetime.datetime.now()
        else:
            self.log = False

    def set_states(self, states):
        sim_states = self.realEnv.get_state()
        sim_states.qpos[:] = states
        self.realEnv.set_state(sim_states)
        self.realEnv.forward()

    def set_Mu_Sigma(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        """This func use random sampling to generate different trajectories for model fitting"""

    def updateDynamics(self):
        """Use the different trajectories to update the linear model"""

    def get_Normal(self, mu, sigma, T = 1):
#         disturb = np.random.normal(0, 0.1, (T, self.num_joint))
        ranVect = np.array(np.transpose([np.random.normal(m, s, T) for m, s in zip(self.mu, np.diag(self.sigma))]))
        return ranVect

    def get_fast_normal(self, T):
        disturb = np.random.normal(0, 5, (T, self.num_joint))
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

    def update_Control(self):
        for i in range(len(self.U)):
            for j in range(len(self.base_control)):
                self.U[i] += self.base_control[j][i] * self.w[j]
        # return U

    def compute_Weight(self):
        lou = min(self.S)
        yita = sum(math.exp((lou - self.S[x])/self.lamb) for x in range(len(self.S)))
        self.w = []
        w_append = self.w.append
        for i in range(len(self.S)):
            w_append(math.exp((lou - self.S[i])/self.lamb)/yita)
#         return self.w

    def apply_Control(self, env, ctrl):
        env.data.ctrl[:] = ctrl
        #env.data.ctrl[:] = np.clip(ctrl, [self.low_bound]*self.num_joint, [self.high_bound]*self.num_joint)
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

    #Multiprocessing but too much
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

            #print(self.S[0])
            self.compute_Weight()
            self.update_Control()

            #Record the states and actions for NN
            self.data["states"].append(np.concatenate([self.realEnv.get_state().qpos, self.realEnv.get_state().qvel]))
            #import pdb; pdb.set_trace()
            self.data["actions"].append(self.U[0])

            self.apply_Control(self.realEnv, self.U[0])
            self.add_U()
            if self.envName == "arm_door":
                wwj_xpos = self.realEnv.data.site_xpos
                #print(wwj_xpos[0], wwj_xpos[2], wwj_xpos[1], wwj_xpos[3])

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                self.record.put(np.flip(self.realEnv.render(1024, 512, device_id = 0), 0))

        #print(time.time()-t1)
        if self.RENDER == "RECORD":
            _fileName = utils.getTimeStamp()
            utils.save_video(self.record, "./videos/video_"+_fileName+".mp4", 10)
            np.savetxt("./data/Real_"+_fileName+".txt", np.asarray(totalCost))
        self.data["states"] = np.asarray(self.data["states"])
        self.data["actions"] = np.asarray(self.data["actions"])
        if self.log:
            f = open("./results.log", "a")
            msg = "K: {}, T: {}, a: {}, lambda: {}, time:{}, cost:{}\n".format(self.K, self.T, self.alpha, self.lamb, (datetime.datetime.now()-self.start_time).seconds, self.S[0])
            f.write(msg)

#         print("Finish MPPI Planning")
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
    parser.add_argument('-eva', '--evaluate', type=bool, default=False,
                        help='if test the task')

    args = parser.parse_args()
    ENV = args.env
    ITER = args.iter

    hm_env_path = "/home/wuweijia/GitHub/MPPI/python/humanoid/humanoid.xml"
    hc_env_path = "/home/wuweijia/GitHub/MPPI/python/half_cheetah/half_cheetah.xml"
    ag_env_path = "/home/wuweijia/GitHub/MPPI/python/arm_gripper/arm_claw.xml"
    ad_env_path = "/home/wuweijia/GitHub/MPPI/python/arm_door/pr2_arm3d_door.xml"

    #ip_args = {"JOINTSNUM":1, "K":20, "T":500, "alpha":0.1, "lamb":0.05, "gama":0.5, "render":"RECORD", "env_path":ip_env, "mu":None, "sigma":None}

    hm_args = {"JOINTSNUM":17, "K":200, "T":100, "alpha":0.1, "lamb":0.5, "gama":0.5, "render":"RECORD", "env_path":hm_env_path, "mu":np.zeros(17), "sigma":0.05*np.eye(17)}

    ag_args = {"JOINTSNUM":9, "K":50, "T":50, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":ag_env_path, "mu":np.zeros(9), "sigma":5*np.eye(9)}

    ad_args = {"JOINTSNUM":6, "K":50, "T":50, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":ad_env_path, "mu":np.zeros(6), "sigma":100*np.eye(6)}

    hc_args = {"JOINTSNUM":6, "K":50, "T":50, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":hc_env_path, "mu":np.zeros(6), "sigma":5*np.eye(6)}

    test_args = {"JOINTSNUM":6, "K":2, "T":2, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":hc_env_path, "mu":np.zeros(6), "sigma":5*np.eye(6), "log":True}

    if args.evaluate:
        import copy
        for param in [ag_args]:
            param["log"] = True
            for name in ["alpha", "lamb"]:
                for time in [0.25, 0.5, 1.0, 2.0, 4.0]:
                    #import pdb; pdb.set_trace()
                    temp_param = copy.deepcopy(param)
                    if name in {"K", "T"}:
                        temp_param[name] = int(param[name]*time)
                    else:
                        temp_param[name] = param[name]*time
                    print(temp_param["JOINTSNUM"], temp_param["K"], temp_param["T"], temp_param["alpha"], temp_param["lamb"], temp_param["gama"])
                    mppi_agent = MPPI(temp_param)
                    mppi_agent.run_MPPI(ITER)
        print("Finish Evaluation.")
        exit()

    args = {"humanoid":hm_args, "robot_arm":ag_args, "half_cheetah":hc_args, "arm_door":ad_args, "test":test_args}


    if ENV not in args:
        print("There is no environment: "+ENV)
    else:
        env_arg = args[ENV]

    mppi_agent = MPPI(env_arg)
    mppi_agent.run_MPPI(ITER)

