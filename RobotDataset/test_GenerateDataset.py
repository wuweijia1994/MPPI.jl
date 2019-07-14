from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py
import h5py
import numpy as np
import logging
import queue
import time
import os
import datetime
import imageio
import argparse

parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.') 
parser.add_argument('mode', help='select one of the following modes: train, valid, test')
mode = parser.parse_args()

logging.basicConfig(level=logging.DEBUG,
                    #filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

envs = {
        "humanoid":{
                "path":"/home/wuweijia/GitHub/MPPI/python/humanoid/humanoid.xml",
                "converge_threshold":0.05,
                "action_dimension":17,},
                #state_dimension = ,},
        "half_cheetah":{
                "path":"/home/wuweijia/GitHub/MPPI/python/half_cheetah/half_cheetah.xml",
                "action_dimension":6,
                "converge_threshold":0.07,},
        "arm_gripper":{
                "path":"/home/wuweijia/GitHub/MPPI/arm_gripper/arm_claw.xml",
                "converge_threshold":0.07,
                "action_dimension":9,}
        }

#env_name = "half_cheetah"
env_name = "humanoid"
#env_name = "arm_gripper"
path = envs[env_name]["path"]
converge_threshold =  envs[env_name]["converge_threshold"]
model = load_model_from_path(path)
real_sim = MjSim(model)
sim_state = real_sim.get_state()
env_dataset = {"state":[], "nextState":[], "time":[], "action":[]}
record = queue.Queue()

if mode.mode == "train":
    dataSize = 10000
    np.random.seed(1331)
elif mode.mode == "valid":
    dataSize = 100
    np.random.seed(1334)
else:
    dataSize = 1000
    np.random.seed(1333)
dataName = mode.mode

#real_viewer = MjViewer(real_sim)
#real_viewer._record_video = True
#real_viewer._render_every_frame = True
#real_viewer._video_idx = 1

def save_video(queue, filename, fps):
    if not os.path.isdir(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))

    writer = imageio.get_writer(filename, fps=fps)
    while not queue.empty():
        frame = queue.get()
        writer.append_data(frame)
    writer.close()

def getNextAction(env_name):
    nextAction = np.random.random((envs[env_name]["action_dimension"],))    
    assert nextAction.shape[0] == envs[env_name]["action_dimension"]
    return nextAction

def recordData(grp, fname, d):
    """the fundamental funcs to copy data"""
    d = np.asarray(d)
    dset = grp.create_dataset(fname, d.shape, dtype='f')
    dset[...] = d

def insertWithTime(grps, datasets):
    """put the data from the env_dataset to the hdf5 dataset with the current time"""
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for name in {"state", "action", "time"}:
        recordData(grps[name], time_now, datasets[name])

def clearTempDataset(datasets):
    """Clear the env_datasets' content after record them into the hdf5 datasets."""
    for name in {"state", "action", "time"}:
        datasets[name]=[]

prev_sim_state = None
f = []
#h5py.File(env_name+"_"+dataName+".hdf5", "w")
#groups = {name:f.create_group(name) for name in {"state", "action", "time"}}

for counter in range(dataSize):
    nextAction = getNextAction(env_name)
    for __ in range(5):
        real_sim.data.ctrl[:] = nextAction
        real_sim.step()
    
    prev_sim_state = sim_state
    sim_state = real_sim.get_state()
    logger.debug("sim vel shape: {}".format(sim_state.qvel.shape))
    logger.debug("sim pos shape: {}".format(sim_state.qpos.shape))
#    print("prev", prev_sim_state)
    #print(prev_sim_state == None)
    if not prev_sim_state == None:
        logger.debug("sim state shape: {}".format(sim_state.qpos.shape))
        env_dataset["state"].append(np.concatenate((sim_state.qpos, sim_state.qvel), axis=None))
        #env_dataset["nextState"].append(np.concatenate((prev_sim_state.qpos, prev_sim_state.qvel), axis=None))
        env_dataset["time"].append(sim_state.time)
        env_dataset["action"].append(nextAction)
    
    record.put(np.flip(real_sim.render(512, 256, device_id = 0), 0))
    #logger.info("sim state: \n{}".format(sim_state))
    if ((counter % 200) == 199):
    #if np.linalg.norm(prev_sim_state.qpos - sim_state.qpos) < converge_threshold:
#        insertWithTime(groups, env_dataset)
#        clearTempDataset(env_dataset)
#        real_sim.reset()
#        sim_state = real_sim.get_state()
#        prev_sim_state = None
        #print("Reset")
        logger.info("Reset simulation")

#with h5py.File(dataName+".hdf5", "w") as f:
#    for key in env_dataset:
#        print(key)
#        #print(env_dataset[key].dtype)
#        env_dataset[key] = np.asarray(env_dataset[key])
#        print(env_dataset[key].shape)
#        dset = f.create_dataset(key, env_dataset[key].shape, dtype="f")
#        dset[...] = env_dataset[key] 

