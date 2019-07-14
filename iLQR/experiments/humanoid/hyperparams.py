""" Hyperparameters for MJC peg insertion trajectory optimization. """
from datetime import datetime
import os.path
import numpy as np

from algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from cost.cost_state import CostState
from cost.cost_action import CostAction
from cost.cost_sum import CostSum

from dynamics.dynamics_lr_prior import DynamicsLRPrior
from dynamics.dynamics_prior_gmm import DynamicsPriorGMM

from traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from policy.lin_gauss_init import init_lqr

from agent.agent import Agent
#from proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
#        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION

SENSOR_DIMS = {
    "JOINT_ANGLES": 24,
    "JOINT_VELOCITIES": 23,
    "ACTION": 17,
}

GAINS = np.ones((SENSOR_DIMS["ACTION"]))

#BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = './experiments/humanoid/'


common = {
#    'experiment_name': 'my_experiment' + '_' + \
#            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
#    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
#    'target_filename': EXP_DIR + 'target.npz',
#    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 14,
}

#if not os.path.exists(common['data_files_dir']):
#    os.makedirs(common['data_files_dir'])

agent = {
    'type': Agent,
    'filename': '/home/wuweijia/GitHub/MPPI/python/humanoid/humanoid.xml',
    'x0': np.zeros(47),
    'dt': 0.025,
    'substeps': 5,
    'conditions': common['conditions'],
    #'pos_body_idx': np.array([1]),
    #'pos_body_offset': [[np.array([0, 0.2, 0])], [np.array([0, 0.1, 0])],
    #                    [np.array([0, -0.1, 0])], [np.array([0, -0.2, 0])]],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': ["JOINT_ANGLES", "JOINT_VELOCITIES"],
    'obs_include': [],
    #'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 10,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / GAINS,
    'init_acc': np.zeros(SENSOR_DIMS["ACTION"]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / GAINS,
}

state_cost = {
    'type': CostState,
    'l1': 1.2,
    'l2': 15,
    'alpha': 1e-5,
    'data_types': {
        'JOINT_ANGLES': {
            'target_state': np.concatenate([np.asarray([0, 0, 1.4]),np.zeros((21,))], axis=-1),  # Target state - must be set.
            'wp': np.concatenate([np.asarray([0, 0, 1]),np.zeros((21,))], axis=-1),  # State weights - must be set.
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost],
    'weights': [0.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 50,
        'min_samples_per_cluster': 60,
        'max_samples': 60,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 50,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

#common['info'] = generate_experiment_info(config)

