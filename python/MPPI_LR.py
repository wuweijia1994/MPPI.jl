from mujoco_py import load_model_from_path, load_model_from_mjb, MjSim, MjViewer
import io
import os
import numpy as np
import math
import copy
import multiprocessing as mp
import utils
import queue
import ModelSampling.iLQRLR as LR
import matplotlib.pyplot as plt
#import threading
np.random.seed(1)
# import matplotlib.pyplot as plt
def flatternStates(data):
    return np.concatenate((data.qpos, data.qvel), axis = 0)

def get_Cost(name, data):
    if name == "half_cheetah":
        state = data
        rootxPos = state[0]
        rootzPos = state[1]
        rootyPos = state[2]

        return (rootxPos-5)**2 + 0.5*(rootyPos)**2

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

        return  (rootzPos-1.4)**2

    else:
        print("no env")
        exit(0)

def get_TerminalCost(name, data):
    if name == "half_cheetah":
        state = data
        rootyPos = state[2]
        if abs(rootyPos)>0.2:
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

def run_Episode_Compare(T, alpha, K, U, Fm, Fv, envState, envPath, output, k, kexi, name="half_cheetah"):
    #_model_state = flatternStates(simEnv.get_state())
    from mujoco_py import load_model_from_path, MjSim
    model = load_model_from_path(envPath)
    simEnv = MjSim(model)
    simEnv.set_state(envState)

    _model_state = flatternStates(envState)
    episode_cost = 0
    _model_episode_cost = 0

    for t in range(T):
        if k < int((1-alpha)*K):
            v = U[t] + kexi[t]
        else:
            v = kexi[t]

        _model_state = Fm[t,:,:].dot(np.hstack([_model_state, np.asarray(v)]))
        _model_episode_cost += get_Cost(name, _model_state)

        simEnv.data.ctrl[:] = v
        simEnv.step()
        state = np.concatenate([simEnv.data.qpos, simEnv.data.qvel], axis=-1)
        episode_cost += get_Cost(name, state)#TODO terminal state cost

    _model_episode_cost += get_TerminalCost(name, _model_state)#TODO define phi, the terminal cost
    episode_cost += get_TerminalCost(name, state)#TODO define phi, the terminal cost
    output.put((k, episode_cost, _model_episode_cost))


def run_Episode(T, alpha, K, U, Fm, Fv, init_state, output, k, kexi, name="half_cheetah"):
    #output.put((k, 1)); return 0
    #simEnv = self.init_SimEnv(realState)
    #import pdb; pdb.set_trace()
    episode_cost = 0
    state = init_state
    states = []
    actions = []

    for t in range(T):
        if k < int((1-alpha)*K):
            v = U[t] + kexi[t]
        else:
            v = kexi[t]

        state = Fm[t,:,:].dot(np.hstack([state, np.asarray(v)]))
        episode_cost += get_Cost(name, state)#TODO terminal state cost

    episode_cost += get_TerminalCost(name, state)#TODO define phi, the terminal cost
    output.put((k, episode_cost))

def simulate(envState, envPath, actions, T, output):
    from mujoco_py import load_model_from_path, MjSim
    model = load_model_from_path(envPath)
    env = MjSim(model)
    env.set_state(envState)

    states = []
    for t in range(T):
        states.append(flatternStates(env.get_state()))
        env.data.ctrl[:] = actions[t, :].tolist()
        env.step()
    #print(t)
    output.put((states))
    #print("good")

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

        self.lr = LR.DynamicsLRPrior()

        #set the env and cost function
        self.set_Cost(args['cost_fun'])
        # print(args['mu'])
        if not args['mu'].any():
            self.set_Mu_Sigma(np.zeros(self.num_joint), 5*np.eye(self.num_joint))
        else:
            self.set_Mu_Sigma(args['mu'], args['sigma'])

#         self.output = mp.Manager().Queue()
        self.output = mp.Queue()
        #self.stateQueue = mp.Queue()
        #self.actionQueue = mp.Queue()
        self.low_bound = -1.0
        self.high_bound = 1.0
        self.init_RealEnv(self.RENDER)
        self.init_U()

        #self.sample(self.realEnv.get_state())
        #self.updateDynamics(self.sampleStateLists, self.sampleActionLists)

    def set_Cost(self, cost_fun = None):
        self.CUSTOM_COST = None
        self.CUSTOM_COST = cost_fun

    def set_Mu_Sigma(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

#     def get_Cost(self, data):
#         if self.CUSTOM_COST:
#             return self.CUSTOM_COST(data)
#         else:
#             printf("There is no customerized cost function.")
#             return sum([x**2] for x in state)
    def sample(self, realState, conditions = 100, T = 50):
        """This func use random sampling to generate different trajectories for model fitting"""
        #import time; _t = time.time();
        self.sampleStateLists = []
        self.sampleActionLists = []
        processes = []
        output = mp.Manager().Queue()
        actions_list=[]
        #actions_list = 2.0 * np.random.random_sample((conditions, T, self.num_joint)) - 1
        #import pdb; pdb.set_trace()
        #for cond in range(conditions):
        #    actions = actions_list[cond, :, :]
        #    simulate(realState, self.env_path, actions, T, output)

        for cond in range(conditions):
            actions = self.get_Normal(self.mu, self.sigma, T)
            actions_list.append(actions)
            processes.append(mp.Process(target=simulate, args=(realState, self.env_path, actions, T, output)))

        #import pdb; pdb.set_trace()
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        #import pdb; pdb.set_trace()
        states_list = [output.get() for k in range(conditions)]
        self.sampleStateLists = np.asarray(states_list)
        self.sampleActionLists = np.asarray(actions_list)
        #print("time", time.time() - _t)

    def updateDynamics(self,sampleStates, sampleActions):
        """Use the different trajectories to update the linear model"""
        #import pdb; pdb.set_trace()
        def normalize(tensor):
            if len(tensor.shape) < 3:
                return tensor[np.newaxis, :]
            else:
                return tensor
        sampleStates = normalize(sampleStates)
        sampleActions = normalize(sampleActions)

        self.lr.update_prior(sampleStates, sampleActions)

    def get_Cost(self, data):
        state = data.site_xpos
        end_pos = state[0]
        obj_pos = state[1]
        target = [0.2, 0.1, 0.2]
        cost = 0
        for i in range(len(end_pos)):
            cost += (end_pos[i]-obj_pos[i])**2
            cost += (target[i]-obj_pos[i])**2
        return cost

    def get_TerminalCost(self, data):
        episode_cost = 0
        state = data.site_xpos
        obj_pos = state[1]
        target = [0.1, 0.1, 0]
        for o,t in zip(obj_pos, target):
            episode_cost += (o-t)**2
        return episode_cost

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
        env.data.ctrl[:] = np.clip(ctrl, [self.low_bound]*self.num_joint, [self.high_bound]*self.num_joint)
        env.step()

    def init_U(self):
        self.U = self.get_Normal(self.mu, self.sigma, self.T)

    def add_U(self):
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.get_Normal(self.mu, self.sigma)

    def run_MPPI(self, iters):
        totalCost = []
        for i in range(iters):#TODO: implement the taskFinish function
            self.lr = LR.DynamicsLRPrior()
            if self.envName == "half_cheetah":
                totalCost.append(get_Cost(self.envName, self.realEnv.data.qpos))
            elif self.envName == "humanoid":
                totalCost.append(get_Cost(self.envName, (self.realEnv.get_state())))
            elif self.envName == "robot_arm":
                totalCost.append(get_Cost(self.envName, (self.realEnv.data)))
            else:
                print("No Env")
                exit(0)
            self.sample(self.realEnv.get_state(), conditions = self.K, T = self.T)
            self.updateDynamics(self.sampleStateLists, self.sampleActionLists)
            self.base_control = copy.deepcopy(self.sampleActionLists)
            #self.sample(self.realEnv.get_state(), conditions = self.K)
            self.lr.fit(self.sampleStateLists, self.sampleActionLists)
            #import pdb; pdb.set_trace()
            self.S=[0]*self.K
            #self.base_control = []
            processes = []
            self.record_RealEnv()
            # self.init_SimEnv()
            _t_output = mp.Queue()

#            for k in range(self.K):
#                #kexi = self.get_Normal(self.mu, self.sigma, self.T)
#                #self.base_control.append(kexi)
#                kexi = self.base_control[k, :, :]
#                _t_simState = self.realEnv.get_state()
#                run_Episode_Compare(self.T, self.alpha, self.K, self.U, self.lr.Fm, self.lr.fv,  _t_simState, self.env_path, _t_output, k, kexi, name="half_cheetah")
#
            for k in range(self.K):
                kexi = self.base_control[k, :, :]
                processes.append(mp.Process(target=run_Episode, args=(self.T, self.alpha, self.K, self.U, self.lr.Fm, self.lr.fv, flatternStates(self.realEnv.get_state()), self.output, k, kexi, "half_cheetah")))
            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()

            results = [self.output.get() for k in range(self.K)]
            results.sort()
            self.S = [r[1] for r in results]

            #results = [_t_output.get() for k in range(self.K)]
            #plt.clf()
            #rr = np.asarray(results)
            #plt.plot(rr[:, 0], rr[:,1], label='real line')
            #plt.plot(rr[:, 0], rr[:,2], label='model line')
            #plt.legend(loc='upper right')
            #plt.savefig("plot.png")

            #_model_S = [r[2] for r in results]
            #_real_S = [r[1] for r in results]

            #def computeW(S):
            #    lou = min(S)
            #    yita = sum(math.exp((lou - S[x])/self.lamb) for x in range(len(S)))
            #    w = []
            #    w_append = w.append
            #    for i in range(len(S)):
            #        w_append(math.exp((lou - S[i])/self.lamb)/yita)
            #    return w

            #def updateU(w, U):
            #    for i in range(len(U)):
            #        for j in range(len(self.base_control)):
            #            U[i] += self.base_control[j][i] * w[j]
            #    return U


            #_model_w = computeW(_model_S)
            #_real_w = computeW(_real_S)

            #plt.clf()
            #plt.plot(range(len(_model_w)), _model_w, label='model line')
            #plt.plot(range(len(_real_w)), _real_w, label='real line')
            #plt.legend(loc='upper right')
            #plt.savefig("plot_w.png")

            #print("model W", _model_w)
            #print("real W", _real_w)


            #_model_U = updateU(_model_w, copy.deepcopy(self.U))
            #_real_U = updateU(_real_w, copy.deepcopy(self.U))

            #plt.clf()
            #plt.plot(range(len(_model_U)), _model_U, label='model line')
            #plt.plot(range(len(_real_U)), _real_U, label='real line')
            #plt.legend(loc='upper right')
            #plt.savefig("plot_U.png")

            #print("model U", _model_U)
            #print("real U", _real_U)
            #import pdb;pdb.set_trace()

            #self.S = _real_S

            print(self.S[0])
            self.compute_Weight()
            self.update_Control()
            self.apply_Control(self.realEnv, self.U[0])
#             print(U[0])
            self.add_U()

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                #import pdb; pdb.set_trace()
                self.record.put(np.flip(self.realEnv.render(1024, 512, device_id = 0), 0))

        if self.RENDER == "RECORD":
            _fileName = utils.getTimeStamp()
            utils.save_video(self.record, "./videos/video_"+utils.getTimeStamp()+".mp4", 10)
            np.savetxt("./data/Model_"+_fileName+".txt", np.asarray(totalCost))
#         print("Finish MPPI Planning")
