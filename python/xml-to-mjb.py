from mujoco_py import load_model_from_path, MjSim, MjViewer
import argparse
from io import BytesIO
path="humanoid.xml"
model = load_model_from_path(path)
model.save("humanoid", format="mjb")
real_sim = MjSim(model)
with BytesIO() as f:
    real_sim.save(f, format="mjb")
