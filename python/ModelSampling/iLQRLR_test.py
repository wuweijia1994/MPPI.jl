import iLQRLR
import numpy as np
import h5py
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import logging
logging.basicConfig(level=logging.DEBUG,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

def draw3D(tensor, env_name):
    """drawa the graph in 3D view"""
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(env_name)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State dimension')
    ax.set_zlabel('Error')
    X = np.arange(tensor.shape[0])
    Y = np.arange(tensor.shape[1])
    np.savetxt("tensor.txt", tensor)
#    print("X", X, "Y", Y)
    X, Y = np.meshgrid(X, Y)
    Z = np.asarray(tensor).T
    
    ax.plot_surface(X, Y, Z)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.savefig("image.png")

def saveData(env_name, y_true, y_pred):
    np.savetxt(env_name+"_y_true.txt", y_true)
    np.savetxt(env_name+"_y_pred.txt", y_pred)

#env_name = "arm_gripper"
env_name = "humanoid"
#env_name = "half_cheetah"
pickle_file_name = './'+env_name+'_lr.pickle'
if os.path.exists(pickle_file_name):
    with open(pickle_file_name,'rb') as r:
        lr = pickle.load(r)
else:
    lr = iLQRLR.DynamicsLRPrior()
    f = h5py.File(env_name+'_train.hdf5','r+')
    groups = {name:f["/"+name] for name in {"state", "action", "time"}}
    print(groups["state"])
    logger.debug("groups[state]: {}".format(groups["state"])) 
    
    for name in groups["state"]:
        #logger.debug("s shape: {}".format(s.shape))
        s = np.asarray(groups["state"][name])[np.newaxis, :]
        a = np.asarray(groups["action"][name])[np.newaxis, :]
        print(s.shape, a.shape)
        lr.update_prior(s, a)
    with open(pickle_file_name,'wb') as p:
        pickle.dump(lr,p)

f = h5py.File(env_name+'_test.hdf5','r+')
groups = {name:f["/"+name] for name in {"state", "action", "time"}}

test_state = []
test_action = []
for name in groups["state"]:
    #print(np.asarray(groups["state"][name]).shape)
    test_state.append(np.asarray(groups["state"][name]))
    test_action.append(np.asarray(groups["action"][name]))
    #a = np.asarray(groups["action"][name])[np.newaxis, :]
test_state = np.asarray(test_state)
test_action = np.asarray(test_action)
print("test state shape", test_state.shape, "test action shape", test_action.shape)
Fm, fv, dyn_covar = lr.fit(test_state, test_action)
print(Fm.shape, fv.shape, dyn_covar.shape)
s_merge =np.concatenate([test_state[3], test_action[3]], axis = 1) 
print(s_merge.shape)
sn = []
for a, b in zip(Fm, s_merge):
    sn.append(np.dot(a, b))
#s_n = np.tensordot(Fm, s_merge, 1)
sn = np.asarray(sn)
sn = sn + fv
print("sn shape", sn.shape)
s_diff = np.abs(sn[:199, :] - test_state[0][1:, :])
saveData(env_name, test_state[0][1:, :],sn[:199, :])

print(s_diff)
draw3D(s_diff[:, :], env_name)
