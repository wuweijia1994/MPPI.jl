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
#     version: 3.6.5
# ---

from mujoco_py import load_model_from_path, MjSim
import MPPI
import copy
import os
import utils
import math
import Memory_Model
# import collections
import numpy as np
import multiprocessing as mp

# MLP for Pima Indians Dataset Serialize to JSON and HDF5


# +
# @profile 
def get_TerminalCost(self, data):
    episode_cost = 0
    state = data.site_xpos
    obj_pos = state[1]
    target = [0.1, 0.1, 0]
    for o,t in zip(obj_pos, target):
        episode_cost += (o-t)**2
    return episode_cost

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

def run_Episode(realState, output, k, kexi, T, K, alpha, U):
    model = load_model_from_path("/home/wuweijia/GitHub/MPPI/python/arm_gripper/arm_claw.xml")
    simEnv = MjSim(model)
    simEnv.set_state(realState)
    
    episode_cost = 0

    for t in range(T):
        if k < int((1-alpha)*K):
            v = U[t] + kexi[t]
        else:
            v = kexi[t]

        simEnv.data.ctrl[:] = v.tolist()
        simEnv.step()
        pos = simEnv.data.qpos

        rootzPos = pos[2]

        episode_cost += (rootzPos-1.4)**2
    episode_cost += self.get_TerminalCost(simEnv.data)
    output.put((k, episode_cost))
#     return (k, episode_cost)
    
    
def run_ag_Episode(realState, output, k, kexi, T, K, alpha, U):
    model = load_model_from_path("/home/wuweijia/GitHub/MPPI/python/arm_gripper/arm_claw.xml")
    simEnv = MjSim(model)
    simEnv.set_state(realState)
    
    episode_cost = 0

    for t in range(T):
        if k < int((1-alpha)*K):
            v = U[t] + kexi[t]
        else:
            v = kexi[t]

        simEnv.data.ctrl[:] = v
        simEnv.step()
        
        state = simEnv.data.site_xpos        

        end_pos = state[0]
        obj_pos = state[1]
        target = [0.2, 0.1, 0.2]
        for i in range(len(end_pos)):
            episode_cost += (end_pos[i]-obj_pos[i])**2
            episode_cost += (target[i]-obj_pos[i])**2
            
    obj_pos = state[1]
    target = [0.1, 0.1, 0]
    for o,t in zip(obj_pos, target):
        episode_cost += (o-t)**2

    output.put((k, episode_cost))    
# -


class MPPI_MMR(MPPI.MPPI):
    """docstring for MPPI."""
    def __init__(self, args):
        super().__init__(args)
        self.state_dimention = len(self.realEnv.get_state().flatten())
        self.model_name = "model"
#         print(np.array(self.U).shape)

    def create_memory(self, RESUME = True):
        self.batch_states = []
        self.batch_actions = []
        self.model=Memory_Model.build_model(RESUME, self.model_name, self.state_dimention, self.num_joint)

    def record_trajectories(self, info, value):
        if info == "state":
            self.batch_states.append(value)
        elif info == "action":
            self.batch_actions.append(value)
        else:
            print("Unrecognized info to record!")

    def build_memory_on_batch(self):
        #model.train_on_batch(x_batch, y_batch)
        if len(self.batch_states) == len(self.batch_actions):
            self.model.train_on_batch(np.asarray(self.batch_states), np.asarray(self.batch_actions))
        else:
            print("The length of batch_states is not equal to batch_actions")

        #No batch data
    def build_memory(self, x, y):
        #model.train_on_batch(x_batch, y_batch)
        x, y = np.asarray(x), np.asarray(y)
        x, y = x[np.newaxis, ...], y[np.newaxis, ...]
        self.model.train_on_batch(x, y)

        #TODO: rsume from the same setting check point
    def save_memory(self):
        Memory_Model.save_model(self.model, self.model_name)

    def train_on_batch(self, input_states, label_traj):
        x, y = np.asarray(input_states), np.asarray(label_traj)
        self.model.fit(x, y, epochs = 2, batch_size = 1)

        #No batch data
    def predict_memory(self, state):
        state = state[np.newaxis, ...]
        return self.model.predict(state)

    def get_expected_trajectory(self, realState):
        traj = [] 
        model = load_model_from_path(self.env_path)
        real_sim = MjSim(model)
        simEnv = self.get_Env()
        simEnv.set_state(realState)
#         simEnv = self.init_SimEnv(realState)

        for i in range(self.T):
            print("simEnv.data.qpos : {}".format(np.array(simEnv.data.qpos)))
            print("simEnv.data.qvel : {}".format(np.array(simEnv.data.qvel)))
            print("simEnv state: {}".format(simEnv.get_state().flatten()))
            v = self.predict_memory(simEnv.get_state().flatten())
#             print("simEnv data : {}".format(simEnv.data.qpos + simEnv.data.qvel))
            v = [i for i in v[0]]

#             print("v: {}".format(v))
            print("v : {}".format(v))
            print("v shape: {}".format(np.array(v).shape))
#             print("v: {}".format(v))
    
            traj.append(v)
#             self.apply_Control(simEnv, v)
            simEnv.data.ctrl[:] = v
#     np.clip(v, [self.low_bound]*self.num_joint, [self.high_bound]*self.num_joint)
            simEnv.step()
        print("traj: {}".format(traj))
        print("network weights: {}".format(self.model.get_weights()))
        print("traj shape: {}".format(np.array(traj).shape))
        return traj

    def get_compute_weight(self, temp_S):
        lou = min(temp_S)
        yita = sum(math.exp((lou - temp_S[x])/self.lamb) for x in range(len(temp_S)))
        temp_w = []
        w_append = temp_w.append
        for i in range(len(temp_S)):
            w_append(math.exp((lou - temp_S[i])/self.lamb)/yita)
        return temp_w
    
    def get_update_control(self, U, base_control, w):
        U, base_control, w = np.array(U), np.array(base_control), np.array(w)
        bias_U = np.average(base_control, axis = 0, weights=w)
        return np.add(U, bias_U).tolist()

#     @profile 
    def get_control_label(self, nn_traj):
        processes = []
        temp_base_control = []
#         print("nn_traj: {}".format(nn_traj))
        tbc_append = temp_base_control.append
        for k in range(self.K):
            temp = self.get_Normal(self.mu, self.sigma, self.T)
#             print("temp control: {}".format(temp))
            kexi = np.add(np.array(nn_traj), temp)
            
            tbc_append(kexi)
            processes.append(mp.Process(target=run_ag_Episode, args=(self.recordRealEnv, self.output, k, kexi, self.T, self.K, self.alpha, self.U)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        results = [self.output.get() for k in range(self.K)]
        results.sort()
        temp_S = [r[1] for r in results]

#         print("S:{}".format(self.S))
        print("temp_S[0]: {}".format(temp_S[0]))
        temp_w = self.get_compute_weight(temp_S)        
        
        processes = []
        return self.get_update_control(self.U, temp_base_control, temp_w)               

#     @profile 
#     def get_control_label(self, nn_traj):
# #         pool = mp.Pool()
#         results = []
#         temp_base_control = []
#         tbc_append = temp_base_control.append
# #         print("nn_traj dimension:{}".format(np.array(nn_traj).shape))
#         for k in range(self.K):
#             temp = self.get_fast_normal(self.T)
#             kexi = np.add(np.array(nn_traj), temp)
#             tbc_append(kexi)
# #             simEnv = self.init_SimEnv(self.recordRealEnv)
# #             processes
#             results.append(run_Episode(self.recordRealEnv, self.output, k, kexi, self.T, self.K, self.alpha, self.U))
#             # Run processes

#         temp_S = [r[1] for r in results]

# #         print("S:{}".format(self.S))
#         temp_w = self.get_compute_weight(temp_S)        
        
#         processes = []
#         return self.get_update_control(self.U, temp_base_control, temp_w) 

    def train_with_label_control(self, label_control, realState):
        simEnv = self.init_SimEnv(realState)
        input_state = []
#         print(label_control)
        for i in range(len(label_control)):
            input_state.append(simEnv.get_state().flatten())        
            self.apply_Control(simEnv, label_control[i])
        input_state = np.array(input_state)
        print("input_state: {}".format(input_state))
        print("label_control: {}".format(label_control))
        self.train_on_batch(input_state, label_control)

    def run_MPPI_GPS_dual(self, iters):
        print("We had better first pre-trained nn")
        self.create_memory(RESUME=True)
        self.iters = iters
        from tqdm import trange
        for i in trange(iters):#TODO: implement the taskFinish function
            self.S=[0]*self.K
            self.base_control = []
            processes = []
            self.record_RealEnv()
            record_state = self.recordRealEnv.flatten()
            # self.init_SimEnv()
            self.record_trajectories("state", record_state)
            for _ in range(1):
                nn_traj = self.get_expected_trajectory(self.recordRealEnv)
                label_traj = self.get_control_label(nn_traj)
                self.train_with_label_control(label_traj, self.recordRealEnv)

            record_action = self.U[0]
            self.record_trajectories("action", self.U[0])
            self.build_memory(record_state, record_action)

            action = self.predict_memory(self.recordRealEnv.flatten())
            self.apply_Control(self.realEnv, action)

            self.add_U()

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                self.record.put(np.flip(self.realEnv.render( 1280, 608, device_id = 0), 0))

        if self.RENDER == "RECORD":
            utils.save_video(self.record, "./videos/video_"+utils.getTimeStamp()+".mp4", 10)

        self.save_memory()
#         print(self.model.test_on_batch(np.asarray(self.batch_states), np.asarray(self.batch_actions)))
        print("Finish MPPI GPS")


    def run_MPPI_GPS(self, iters):
        print("We had better first pre-trained nn")
        self.create_memory(RESUME=True)
        self.iters = iters
        from tqdm import trange
        for i in trange(iters):#TODO: implement the taskFinish function
            self.S=[0]*self.K
            self.base_control = []
            processes = []
            self.record_RealEnv()
            record_state = self.recordRealEnv.flatten()
            # self.init_SimEnv()
            self.record_trajectories("state", record_state)

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

            record_action = self.U[0]
            self.record_trajectories("action", self.U[0])
            self.build_memory(record_state, record_action)

            action = self.predict_memory(self.recordRealEnv.flatten())
            self.apply_Control(self.realEnv, action)

            self.add_U()

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                self.record.put(np.flip(self.realEnv.render( 1280, 608, device_id = 0), 0))

        if self.RENDER == "RECORD":
            utils.save_video(self.record, "./videos/video_"+utils.getTimeStamp()+".mp4", 10)

        self.build_memory_on_batch()
        self.save_memory()
#         print(self.model.test_on_batch(np.asarray(self.batch_states), np.asarray(self.batch_actions)))
        print("Finish MPPI GPS")

    def run_MPPI_GPS_with_pool(self, iters):
        print("We had better first pre-trained nn")
        self.create_memory(RESUME=True)
        self.iters = iters

        from tqdm import tqdm
        for i in tqdm(range(iters)):#TODO: implement the taskFinish function
            self.S=[0]*self.K
            self.base_control = []

            results = [1]*self.K

            pool = mp.Pool(processes=12)
            self.record_RealEnv()
            record_state = self.recordRealEnv.flatten()
            self.record_trajectories("state", record_state)

            for k in range(self.K):
                kexi = self.get_Normal(self.mu, self.sigma, self.T)
                self.base_control.append(kexi)
                pool.apply_async(self.run_Episode_with_return, args=(self.recordRealEnv, self.output, k, kexi))
            pool.close()
            pool.join()
#                results.append(1)

            self.S = results
#             print("get results")
            self.compute_Weight()
            self.update_Control()

            record_action = self.U[0]
            self.record_trajectories("action", self.U[0])
            self.build_memory(record_state, record_action)

            action = self.predict_memory(self.recordRealEnv.flatten())
            self.apply_Control(self.realEnv, action)

            self.add_U()

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                self.record.put(np.flip(self.realEnv.render( 1280, 608, device_id = 0), 0))

        if self.RENDER == "RECORD":
            utils.save_video(self.record, "./videos/video_"+utils.getTimeStamp()+".mp4", 10)

        self.build_memory_on_batch()
        self.save_memory()
#         print(self.model.test_on_batch(np.asarray(self.batch_states), np.asarray(self.batch_actions)))
        print("Finish MPPI GPS")

    def run_MPPI_Supervising(self, iters):
        self.create_memory(RESUME=True)
        self.iters = iters

        for i in range(iters):#TODO: implement the taskFinish function
            self.S=[0]*self.K
            self.base_control = []
            processes = []
            self.record_RealEnv()
            self.record_trajectories("state", self.recordRealEnv.flatten())
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
            self.record_trajectories("action", self.U[0])

            self.add_U()

            if self.RENDER=="RENDER":
                self.CUSTOM_VIEWER.render()

            elif self.RENDER == "RECORD":
                self.record.put(np.flip(self.realEnv.render( 1280, 608, device_id = 0), 0))

        if self.RENDER == "RECORD":
            utils.save_video(self.record, "./videos/video_"+utils.getTimeStamp()+".mp4", 10)

        self.build_memory()
        self.save_memory()
        print(self.model.test_on_batch(np.asarray(self.batch_states), np.asarray(self.batch_actions)))
        print("Finish MPPI supervising")








