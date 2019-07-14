
from mujoco_py import load_model_from_path, MjSim, MjViewer

model = load_model_from_path("humanoid.xml")
mj_sim = MjSim(model)
print(mj_sim.get_state())

hello_model = load_model_from_path("hello.xml")
hello_mj_sim = MjSim(hello_model)
print(hello_mj_sim.data.ctrl)
for _ in range(100):
    print(hello_mj_sim.get_state().qvel)
    hello_mj_sim.step()
