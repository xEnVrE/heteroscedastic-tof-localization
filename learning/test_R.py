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
import matplotlib.pyplot as plt
import numpy
import pickle
import torch
import torchfilter
from filter.motion_model import MotionModel
from filter.measurement_model import MeasurementModel
# from filter.feature_kalman_filter import FeatureExtendedKalmanFilter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from simulator.sim import Simulator
from simulator.viewer import Viewer
from tqdm import tqdm


class Tester():

    def __init__(self, options):
        """Constuctor."""

        torch.manual_seed(0)
        torch.set_deterministic(True)
        numpy.random.seed(0)

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

        # Setup matplotlib
        font_size = 16
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("$x_1$", fontsize = font_size + 10)
        ax.set_ylabel("$x_2$", fontsize = font_size + 10)
        ax.set_zlabel("$R$", fontsize = font_size + 10)
        ax.xaxis.set_tick_params(labelsize = 10)

        x = []
        y = []
        z = []
        z_alt = []
        with torch.no_grad():
            for i in numpy.linspace(0.0, 2.0, 100):
                for j in numpy.linspace(0.0, 2.0, 100):
                    x.append(i)
                    y.append(j)

                    state = torch.Tensor([i, j, 0.0, 0.0])
                    state = state[None, :]
                    R = torch.diag_embed(torch.exp(self.filter_model.measurement_model.R(self.filter_model.measurement_model.state_feature(state / 2))))[0].numpy()
                    # R = self.filter_model.measurement_model.R.numpy()
                    R = R.dot(R.T)
                    R_diag = numpy.diag(R)
                    z.append(R_diag[0])
                    z_alt.append(0.077)

        ax.scatter(x, y, z, s = 2, c = 'C1')
        # ax.scatter(x, y, z_alt, s = 2, c = 'C2')
        cf = plt.gcf()
        cf.set_size_inches((12, 12))
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', dest = 'dataset_path', type = str, required = True)

    options = parser.parse_args()


    tester = Tester(options)
    # for i in range(len(tester.test_set)):
    #     time_history, robot_history, sensors_history, estimate_history = tester.test(i, use_train_set = False)

    #     error_x = estimate_history[:, 0] - robot_history[:, 0]
    #     error_y = estimate_history[:, 1] - robot_history[:, 1]
    #     rmse_x_i = numpy.linalg.norm(error_x) / numpy.sqrt(error_x.shape[0])
    #     rmse_y_i = numpy.linalg.norm(error_y) / numpy.sqrt(error_y.shape[0])
    #     rmse_x.append(rmse_x_i)
    #     rmse_y.append(rmse_y_i)

    #     max_x.append(numpy.max(abs(error_x)))
    #     max_y.append(numpy.max(abs(error_y)))

    #     v = Viewer(limit, s.sensors_positions())
    #     # v.show(time_history, history_state = robot_history, history_sensors = sensors_history, history_estimate = estimate_history, save_to = './debug/' + str(i) + '.png')
    #     v.show(time_history, history_state = robot_history, history_sensors = sensors_history, history_estimate = estimate_history)

if __name__ == '__main__':
    main()
