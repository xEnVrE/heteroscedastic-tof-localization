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
import numpy
import pickle
import torchfilter
from simulator.sim import Simulator
from torchfilter import types
from tqdm import tqdm


class Dataset():

    def __init__(self, options):
        """Constructor."""

        self.options = options

        # Instantiate the simulator
        self.dt = 1.0 / 30.0
        psd = 0.1
        limit = 2.0
        self.sim = Simulator(dt = self.dt, psd = psd, limit = limit)

        # Storage for trajectories
        self.test_set_ids = [i for i in range(self.options.number_experiments)][::self.options.testing_stride]
        self.test_set = []
        self.train_set = []


    def generate(self):
        """Generate dataset for testing and training."""

        for i in tqdm(range(self.options.number_experiments)):

            self.sim.reset()

            experiment_length = self.options.experiment_length
            if i in self.test_set_ids:
                experiment_length = 10.0

            hist_u = []
            for j in range(int(experiment_length / self.dt)):
                self.sim.step()
                hist_u.append([0.0])

            hist_x = self.sim.robot_history()
            hist_y = self.sim.sensors_history()
            hist_u = numpy.array(hist_u)

            trajectory = types.TrajectoryNumpy(states = hist_x, observations = hist_y, controls = hist_u)
            if i in self.test_set_ids:
                self.test_set.append(trajectory)
            else:
                self.train_set.append(trajectory)

        print('# train set: ' +  str(len(self.train_set)) + ' trajectories')
        print('# test set: ' + str(len(self.test_set)) + ' trajectories')

        return self.train_set, self.test_set


    def save(self):
        """Save the dataset."""

        output_file_name = './dataset.pickle'

        with open(output_file_name, 'wb') as output:
            pickle.dump({'train_set' : self.train_set, 'test_set' : self.test_set}, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-length', dest = 'experiment_length', type = float, required = False, default = 200.0)
    parser.add_argument('--number-experiments', dest = 'number_experiments', type = int, required = False, default = 100)
    parser.add_argument('--testing-stride', dest = 'testing_stride', type = int, required = False, default = 2)
    parser.add_argument('--save', dest = 'save', type = bool)

    options = parser.parse_args()

    dataset = Dataset(options)
    dataset.generate()
    if options.save:
        dataset.save()

if __name__ == '__main__':
    main()
