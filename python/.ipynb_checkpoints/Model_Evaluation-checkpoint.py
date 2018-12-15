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

import os
import utils
import queue
import numpy as np
import Memory_Model as mm
from mujoco_py import load_model_from_path, MjSim, MjViewer

def hm_env(path=os.path.join(os.curdir, "humanoid/humanoid.xml")):
    model = load_model_from_path(path)
    real_sim = MjSim(model)
    return real_sim

model = mm.load_model("./model")
env = hm_env()
record=queue.Queue()
for i in range(500):
    record.put(np.flip(env.render( 1280, 608, device_id = 0), 0))
    curr_state = env.get_state().flatten()
    curr_state = curr_state[np.newaxis, ...]
    act = model.predict(curr_state)
    env.data.ctrl[:] = act
    env.step()
utils.save_video(record, "./videos/video_"+utils.getTimeStamp()+".mp4", 10)


