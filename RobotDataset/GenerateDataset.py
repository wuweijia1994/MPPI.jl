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

logging.basicConfig(level=logging.INFO,
                    filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

envs = {
        "humanoid":{
                "path":"/home/wuweijia/GitHub/MPPI/python/humanoid/humanoid.xml",
                "action_dimension":17,}}
                #state_dimension = ,},
        #"half_cheetah":{}}

env_name = "humanoid"
path = envs[env_name]["path"]
model = load_model_from_path(path)
real_sim = MjSim(model)
sim_state = real_sim.get_state()
env_dataset = {"qpos":[], "qvel":[], "time":[], "action":[]}
record = queue.Queue()

if mode.mode == "train":
    dataSize = 10000
    np.random.seed(1331)
elif mode.mode == "valid":
    dataSize = 100
    np.random.seed(1332)
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


for _ in range(dataSize):
    nextAction = getNextAction(env_name)
    real_sim.data.ctrl[:] = nextAction
    real_sim.step()
    prev_sim_state = sim_state
    sim_state = real_sim.get_state()
    env_dataset["qpos"].append(sim_state.qpos)
    env_dataset["qvel"].append(sim_state.qvel)
    env_dataset["time"].append(sim_state.time)
    env_dataset["action"].append(nextAction)
    
    record.put(np.flip(real_sim.render(1280, 608, device_id = 0), 0))
    logger.info("sim state: \n{}".format(sim_state))
    if np.linalg.norm(prev_sim_state.qpos - sim_state.qpos) < 0.1**3*2:
        real_sim.reset()
        sim_state = real_sim.get_state()
        print("Reset")
        logger.info("Reset simulation")
#save_video(record, "./videos.mp4", 10)

with h5py.File(dataName+".hdf5", "w") as f:
    for key in env_dataset:
        print(key)
        #print(env_dataset[key].dtype)
        env_dataset[key] = np.asarray(env_dataset[key])
        print(env_dataset[key].shape)
        dset = f.create_dataset(key, env_dataset[key].shape, dtype="f")
        dset[...] = env_dataset[key] 

