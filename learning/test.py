#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import argparse
import copy
import fannypack
import numpy
import pickle
import torch
import torchfilter
from filter.motion_model import MotionModel
from filter.measurement_model import MeasurementModel
# from filter.feature_kalman_filter import FeatureExtendedKalmanFilter
from simulator.sim import Simulator
from simulator.viewer import Viewer
from tqdm import tqdm


class Tester():

    def __init__(self, options):
        """Constuctor."""

        torch.manual_seed(0)
        torch.set_deterministic(True)
        numpy.random.seed(0)

        # Load testing dataset
        f = open(options.dataset_path, 'rb')
        self.data = pickle.load(f)
        f.close()

        self.test_set = self.data['test_set']

        # Folders for input
        checkpoint_dir = './output_data/checkpoint'
        metadata_dir = './output_data/metadata'
        log_dir = './output_data/log'

        # Setup training helper
        self.buddy = fannypack.utils.Buddy\
        (
            experiment_name = 'train',
            checkpoint_dir = checkpoint_dir,
            metadata_dir = metadata_dir,
            log_dir = log_dir,
            optimizer_checkpoint_interval = 0,
            cpu_only = True
        )

        # Filter initialization
        self.dt = 1.0 / 30.0
        self.motion_model = MotionModel(self.dt, 1.0).to(device = self.buddy.device)
        self.measurement_model = MeasurementModel(0.1) .to(device = self.buddy.device)
        # self.filter_model = FeatureExtendedKalmanFilter\
        self.filter_model = torchfilter.filters.ExtendedKalmanFilter\
        (
            dynamics_model = self.motion_model,
            measurement_model = self.measurement_model
        # ).to(device = 'cpu')
        ).to(device = self.buddy.device)
        self.buddy.attach_model(self.filter_model)

        self.buddy.load_checkpoint(label = 'sub_16')
        self.filter_model.eval()
        # with torch.no_grad():
        #     self.filter_model.measurement_model.R *= 100.0
        #     R = self.filter_model.measurement_model.R.numpy()
        #     R = R.dot(R.T)
        #     print(R)
        #     # Make sure that R is positive definite
        #     numpy.linalg.cholesky(R)


    def test(self, trajectory_index, initial_cov_scale = 0.01, use_train_set = False, plot = False):
        """Test a given trajectory."""

        if use_train_set:
            trajectory = self.train_set[trajectory_index]
        else:
            trajectory = self.test_set[trajectory_index]

        states = trajectory.states
        measurements = trajectory.observations
        controls = trajectory.controls

        state_dim = self.filter_model.dynamics_model.state_dim
        measurement_dim = self.filter_model.measurement_model.measurement_dim

        times = []
        predicted = []
        corrected = []
        R = []

        with torch.no_grad():

            initial_state = (torch.from_numpy(states[0]).to(device = self.buddy.device))[None, :].expand(1, state_dim)
            initial_covariance = (torch.eye(state_dim, device = self.buddy.device) * initial_cov_scale)[None, :, :].expand(1, state_dim, state_dim)

            self.filter_model.initialize_beliefs(mean = initial_state, covariance = initial_covariance)

            for i in tqdm(range(len(measurements))):

                m = (torch.from_numpy(measurements[i]).to(device = self.buddy.device))[None, :].expand(1, measurement_dim)
                c = (torch.from_numpy(controls[i]).to(device = self.buddy.device))[None, :]

                self.filter_model._predict_step(controls = c)
                belief_predicted = copy.deepcopy(self.filter_model.belief_mean.numpy())

                r = torch.diag_embed(torch.exp(self.filter_model.measurement_model.R(self.filter_model.measurement_model.state_feature(self.filter_model.belief_mean / 2))))[0].numpy()
                r = r.dot(r.T)
                r_diag = numpy.diag(r)
                R.append(r_diag)

                self.filter_model._update_step(observations = m)
                belief_corrected = copy.deepcopy(self.filter_model.belief_mean.numpy())

                predicted.append(belief_predicted[0])
                corrected.append(belief_corrected[0])

                times.append(i * self.dt)

        times = numpy.array(times)
        predicted = numpy.array(predicted)
        corrected = numpy.array(corrected)
        R = numpy.array(R)

        return times, states, measurements, corrected, R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', dest = 'dataset_path', type = str, required = True)

    dt = 1.0 / 30.0
    psd = 0.1
    limit = 2.0
    s = Simulator(dt = dt, psd = psd, limit = limit)

    options = parser.parse_args()

    rmse_x = []
    rmse_y = []
    max_x = []
    max_y = []

    tester = Tester(options)
    for i in range(len(tester.test_set)):
        time_history, robot_history, sensors_history, estimate_history, R_history = tester.test(i, use_train_set = False)

        error_x = estimate_history[:, 0] - robot_history[:, 0]
        error_y = estimate_history[:, 1] - robot_history[:, 1]
        rmse_x_i = numpy.linalg.norm(error_x) / numpy.sqrt(error_x.shape[0])
        rmse_y_i = numpy.linalg.norm(error_y) / numpy.sqrt(error_y.shape[0])
        rmse_x.append(rmse_x_i)
        rmse_y.append(rmse_y_i)

        max_x.append(numpy.max(abs(error_x)))
        max_y.append(numpy.max(abs(error_y)))

        v = Viewer(limit, s.sensors_positions())
        # v.show(time_history, history_state = robot_history, history_sensors = sensors_history, history_estimate = estimate_history, save_to = './debug/' + str(i) + '.png')
        v.show(time_history, history_state = robot_history, history_sensors = sensors_history, history_estimate = estimate_history, history_R = R_history)

    print('RMSE x (max x):')
    print(str(numpy.mean(rmse_x)) + '(' + str(numpy.mean(max_x)) + ')')

    print('RMSE y (max y):')
    print(str(numpy.mean(rmse_y)) + '(' + str(numpy.mean(max_y)) + ')')

if __name__ == '__main__':
    main()
