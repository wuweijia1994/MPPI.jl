import numpy as np
import h5py as h5
import sklearn.metrics  as mt

env_name = "humanoid"
y_true = np.loadtxt(env_name+"_y_true.txt").astype(float)
y_pred = np.loadtxt(env_name+"_y_pred.txt").astype(float)
 
for method in {mt.mean_squared_error, mt.mean_squared_log_error}:
    output = method(y_true, y_pred, multioutput="raw_values")
    import pdb; pdb.set_trace()
