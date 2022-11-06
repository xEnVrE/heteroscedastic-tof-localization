#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import matplotlib.pyplot as plt
import numpy
import pickle
import torch
from filter.motion_model import MotionModel
from filter.measurement_model import MeasurementModel
from filter.ekf import ExtendedKalmanFilter
# from mpl_toolkits.mplot3d import Axes3D


class Tester():

    def __init__(self):
        """Constuctor."""

        torch.manual_seed(0)
        numpy.random.seed(0)
        device = torch.device('cpu')

        # Folders for input
        checkpoint_dir = './train_06_11_2022_11_07_05/0004.pth'
        data = torch.load(checkpoint_dir)

        # Filter initialization
        self.dt = 1.0 / 30.0
        self.motion_model = MotionModel(self.dt, 1.0, device = device).to(device = device)
        self.measurement_model = MeasurementModel(0.1, 32, device = device) .to(device = device)
        self.filter_model = ExtendedKalmanFilter\
        (
            motion_model = self.motion_model,
            measurement_model = self.measurement_model
        ).to(device = device)
        self.filter_model.load_state_dict(data['filter'])
        self.filter_model.eval()

        # Setup matplotlib
        font_size = 16
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
                    R = torch.diag_embed(self.filter_model.measurement_model.R(self.filter_model.measurement_model.state_feature(state / 2)))[0].numpy()
                    R = R.dot(R.T)
                    R_diag = numpy.diag(R)
                    z.append(R_diag[0])
                    z_alt.append(0.077)

        ax.scatter(x, y, z, s = 2, c = 'C1')
        cf = plt.gcf()
        cf.set_size_inches((12, 12))
        plt.savefig('./output.png', dpi = 300)


def main():
    tester = Tester()

if __name__ == '__main__':
    main()
