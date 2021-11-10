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
from matplotlib import rc


class Viewer():

    def __init__(self, limit, sensors_positions):
        """Constructor."""

        self.limit = limit
        self.sensors_positions = sensors_positions

        # Setup matplotlib
        self.font_size = 16
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)
        plt.rc('xtick',labelsize = self.font_size)
        plt.rc('ytick',labelsize = self.font_size)


    def show(self, history_time, history_state = None, history_sensors = None, history_sensors_gt = None, history_estimate = None, history_R = None, save_to = ''):
        """Show the robot position, the robot velocity and the readings from the sensors."""

        if history_state is not None:
            # Robot position
            fig, ax = plt.subplots(1)
            ax.plot(history_state[:, 0], history_state[:, 1], c = 'C0')
            if history_estimate is not None:
                ax.plot(history_estimate[:, 0], history_estimate[:, 1], c = 'C6')

            for i in range(len(self.sensors_positions)):
                position = self.sensors_positions[i]
                ax.scatter([position[0]], [position[1]], s = 200, c = 'C' + str(i + 1), marker = '*')

            border = 0.2
            ax.set_xlim(-border + 0.0, border + self.limit)
            ax.set_ylim(-border + 0.0, border + self.limit)
            ax.axes.set_aspect('equal')
            ax.set_xlabel('$x_1\,(m)$', fontsize = self.font_size)
            ax.set_ylabel('$x_2\,(m)$', fontsize = self.font_size)
            ax.grid()
            cf = plt.gcf()
            cf.set_size_inches((8, 8))

            legend_items = ["$\mathrm{Ground truth}$", "$\mathrm{Estimate}$"]
            fig.legend(labels = legend_items, ncol = 2, loc = 'upper center', frameon = False, fontsize = self.font_size)

            if save_to:
                plt.savefig(save_to, bbox_inches='tight', dpi = 300)

            # Robot velocity
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(history_time, history_state[:, 2])
            ax[1].plot(history_time, history_state[:, 3])
            if history_estimate is not None:
                ax[0].plot(history_time, history_estimate[:, 2], c = 'C6')
                ax[1].plot(history_time, history_estimate[:, 3], c = 'C6')
            ax[0].set_title("$\dot x$", fontsize = self.font_size)
            ax[1].set_title("$\dot y$", fontsize = self.font_size)
            for i in range(2):
                ax[i].grid()
                ax[i].set_ylabel("$(m/s)$", fontsize = self.font_size)

            plt.subplots_adjust(hspace = 0.6)
            cf = plt.gcf()
            cf.set_size_inches((6, 6))

        # Sensor readings
        if history_sensors is not None:
            fig, ax = plt.subplots(2, 2)
            for i in range(history_sensors.shape[1]):
                j = int(i / 2)
                k = int(i % 2)
                axes = ax[j, k]
                # axes.plot(history_time, history_sensors[:, i], c = 'C' + str(i + 1))
                axes.plot(history_time, history_R[:, i], c = 'C' + str(i + 1))
                if history_sensors_gt is not None:
                    axes.plot(history_time, history_sensors_gt[:, i], c = 'grey')
                axes.grid()
                axes.set_xlabel('$t(s)$', fontsize = self.font_size)
                # axes.set_ylabel('$(m)$', fontsize = self.font_size)
                axes.set_ylim(0.0, 0.5)
                # axes.set_title('$d_{' + str(i + 1) +  '}$', fontsize = self.font_size)
                axes.set_title('$R_{' + str(i + 1) +  '}$', fontsize = self.font_size)

            plt.subplots_adjust(hspace = 0.6)
            cf = plt.gcf()
            cf.set_size_inches((12, 6))

        if not save_to:
            plt.show()
