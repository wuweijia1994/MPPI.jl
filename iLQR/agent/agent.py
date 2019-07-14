""" This file defines the base agent class. """
import os
import imageio
import time
import datetime
import abc
import copy
import queue
import numpy as np

from agent.config import AGENT
#from gps.proto.gps_pb2 import ACTION
from agent.agent_utils import generate_noise, setup
from sample.sample_list import SampleList
from sample.sample import Sample
from mujoco_py import load_model_from_path, MjSim, MjViewer

class Agent(object):
    """
    Agent superclass. The agent interacts with the environment to
    collect samples.
    """
    #__metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams = config

        # Store samples, along with size/index information for samples.
        self._samples = [[] for _ in range(self._hyperparams['conditions'])]
        self.T = self._hyperparams['T']
        self.dU = self._hyperparams['sensor_dims']["ACTION"]

        self.x_data_types = self._hyperparams['state_include']
        self.obs_data_types = self._hyperparams['obs_include']
        if 'meta_include' in self._hyperparams:
            self.meta_data_types = self._hyperparams['meta_include']
        else:
            self.meta_data_types = []

        # List of indices for each data type in state X.
        self._state_idx, i = [], 0
        for sensor in self.x_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._state_idx.append(list(range(i, i+dim)))
            i += dim
        self.dX = i

        # List of indices for each data type in observation.
        self._obs_idx, i = [], 0
        for sensor in self.obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dO = i

        # List of indices for each data type in meta data.
        self._meta_idx, i = [], 0
        for sensor in self.meta_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._meta_idx.append(list(range(i, i+dim)))
            i += dim
        self.dM = i



        self._x_data_idx = {d: i for d, i in zip(self.x_data_types,
                                                 self._state_idx)}
        self._obs_data_idx = {d: i for d, i in zip(self.obs_data_types,
                                                   self._obs_idx)}
        self._meta_data_idx = {d: i for d, i in zip(self.meta_data_types,
                                                   self._meta_idx)}

        self._setup_conditions()
        self._setup_world(hyperparams['filename'])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        pass

    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        #import pdb; pdb.set_trace()
        self._world = []
        self._model = []

        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        #import pdb; pdb.set_trace()
        model = load_model_from_path(filename)
        world = MjSim(model)

        self._world = [world for _ in range(self._hyperparams['conditions'])]
        self._model = [model for _ in range(self._hyperparams['conditions'])]

#        for i in range(self._hyperparams['conditions']):
#            for j in range(len(self._hyperparams['pos_body_idx'][i])):
#                idx = self._hyperparams['pos_body_idx'][i][j]
#                self._model[i]['body_pos'][idx, :] += \
#                        self._hyperparams['pos_body_offset'][i][j]

        #self._joint_idx = list(range(self._model[0]['nq']))
        #self._vel_idx = [i + self._model[0]['nq'] for i in self._joint_idx]

        self.x0 = self._hyperparams["x0"]

#        # Initialize x0.
#        self.x0 = []
#        for i in range(self._hyperparams['conditions']):
#            if END_EFFECTOR_POINTS in self.x_data_types:
#                # TODO: this assumes END_EFFECTOR_VELOCITIES is also in datapoints right?
#                self._init(i)
#                eepts = self._world[i].get_data()['site_xpos'].flatten()
#                self.x0.append(
#                    np.concatenate([self._hyperparams['x0'][i], eepts, np.zeros_like(eepts)])
#                )
#            elif END_EFFECTOR_POINTS_NO_TARGET in self.x_data_types:
#                self._init(i)
#                eepts = self._world[i].get_data()['site_xpos'].flatten()
#                eepts_notgt = np.delete(eepts, self._hyperparams['target_idx'])
#                self.x0.append(
#                    np.concatenate([self._hyperparams['x0'][i], eepts_notgt, np.zeros_like(eepts_notgt)])
#                )
#            else:
#                self.x0.append(self._hyperparams['x0'][i])
#            if IMAGE_FEAT in self.x_data_types:
#                self.x0[i] = np.concatenate([self.x0[i], np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],))])
#
#        cam_pos = self._hyperparams['camera_pos']
#        for i in range(self._hyperparams['conditions']):
#            self._world[i].init_viewer(AGENT_MUJOCO['image_width'],
#                                       AGENT_MUJOCO['image_height'],
#                                       cam_pos[0], cam_pos[1], cam_pos[2],
#                                       cam_pos[3], cam_pos[4], cam_pos[5])


    def sample(self, policy, condition, verbose=False, save=False, noisy=False):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features
        if save:
            record = queue.Queue()
        new_sample = self._init_sample(condition, feature_fn=feature_fn)
        #mj_X = self.x0
        #import pdb; pdb.set_trace()
        #self._world[condition].set_state_from_flattened(mj_X)
        self._world[condition].reset()
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self._world[condition].data.ctrl[:] = mj_U
                    self._world[condition].step()

                #import pdb; pdb.set_trace()

                self._data = self._world[condition].data
                #sim_state = self._world[condition].get_state()
                #mj_X = np.concatenate((sim_state.qpos, sim_state.qvel), axis=None)
                if save:
                    record.put(np.flip(self._world[condition].render(512, 256, device_id = 0), 0))
                self._set_sample(new_sample, self._data, t, condition, feature_fn=feature_fn)
        new_sample.set("ACTION", U)
        new_sample.set("NOISE", noise)
        if save:
            def save_video(queue, filename, fps):
                if not os.path.isdir(os.path.dirname(filename)):
                    os.mkdir(os.path.dirname(filename))

                writer = imageio.get_writer(filename, fps=fps)
                while not queue.empty():
                    frame = queue.get()
                    writer.append_data(frame)
                writer.close()

            save_video(record, "./videos.mp4", 10)
        #if save:
        self._samples[condition].append(new_sample)
        return new_sample

    def reset(self, condition):
        """ Reset environment to the specified condition. """
        for i in range(self._hyperparams['conditions']):
            self._world[i].reset()
        #pass  # May be overridden in subclass.

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: funciton to comptue image features from the observation.
        """
        sample = Sample(self)

        # Initialize sample with stuff from _data
        data = self._world[condition].data
        for sensor_name in self._hyperparams["state_include"]:
            if sensor_name == "JOINT_ANGLES":
                sample.set(sensor_name, data.qpos, t=0)
            elif sensor_name == "JOINT_VELOCITIES":
                sample.set(sensor_name, data.qvel, t=0)
            elif sensor_name == "END_EFFECTOR_POINTS":
                sample.set(sensor_name, data.site_xpos, t=0)
            elif sensor_name == "END_EFFECTOR_POINT_VELOCITIES":
                sample.set(sensor_name, np.zeros_like(data.site_xpos), t=0)

        return sample


    def get_samples(self, condition, start=0, end=None):

        """
        Return the requested samples based on the start and end indices.
        Args:
            start: Starting index of samples to return.
            end: End index of samples to return.
        """
        #import pdb; pdb.set_trace()
        return (SampleList(self._samples[condition][start:]) if end is None
                else SampleList(self._samples[condition][start:end]))

    def _set_sample(self, sample, data, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """
        for sensor_name in self._hyperparams["state_include"]:
            if sensor_name == "JOINT_ANGLES":
                sample.set(sensor_name, data.qpos, t=t+1)
            elif sensor_name == "JOINT_VELOCITIES":
                sample.set(sensor_name, data.qvel, t=t+1)
            elif sensor_name == "END_EFFECTOR_POINTS":
                sample.set(sensor_name, data.site_xpos, t=t+1)
            elif sensor_name == "END_EFFECTOR_POINT_VELOCITIES":
                cur_eepts = data.site_xpos
                prev_eepts = sample.get("END_EFFECTOR_POINTS", t=t)
                eept_vels = (cur_eepts - prev_eepts) / self._hyperparams['dt']
                sample.set(sensor_name, eept_vels, t=t+1)

    def clear_samples(self, condition=None):
        """
        Reset the samples for a given condition, defaulting to all conditions.
        Args:
            condition: Condition for which to reset samples.
        """
        if condition is None:
            self._samples = [[] for _ in range(self._hyperparams['conditions'])]
        else:
            self._samples[condition] = []

    def delete_last_sample(self, condition):
        """ Delete the last sample from the specified condition. """
        self._samples[condition].pop()

    def get_idx_x(self, sensor_name):
        """
        Return the indices corresponding to a certain state sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._x_data_idx[sensor_name]

    def get_idx_obs(self, sensor_name):
        """
        Return the indices corresponding to a certain observation sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._obs_data_idx[sensor_name]

    def pack_data_obs(self, existing_mat, data_to_insert, data_types,
                      axes=None):
        """
        Update the observation matrix with new data.
        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dO:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dO)
            insert_shape[axes[i]] = len(self._obs_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._obs_data_idx[data_types[i]][0],
                                   self._obs_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def pack_data_meta(self, existing_mat, data_to_insert, data_types,
                       axes=None):
        """
        Update the meta data matrix with new data.
        Args:
            existing_mat: Current meta data matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dM:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dM)
            insert_shape[axes[i]] = len(self._meta_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._meta_data_idx[data_types[i]][0],
                                   self._meta_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def pack_data_x(self, existing_mat, data_to_insert, data_types, axes=None):
        """
        Update the state matrix with new data.
        Args:
            existing_mat: Current state matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)
            insert_shape[axes[i]] = len(self._x_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def unpack_data_x(self, existing_mat, data_types, axes=None):
        """
        Returns the requested data from the state matrix.
        Args:
            existing_mat: State matrix to unpack from.
            data_types: Names of the sensor to unpack.
            axes: Which axes to unpack along. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        return existing_mat[index]
