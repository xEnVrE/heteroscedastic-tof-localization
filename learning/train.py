#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import argparse
import fannypack
import pickle
import os
import torch
import torchfilter
from filter.motion_model import MotionModel
from filter.measurement_model import MeasurementModel
# from filter.feature_kalman_filter import FeatureExtendedKalmanFilter


class Trainer():

    def __init__(self, options):
        """Constructor."""

        torch.manual_seed(0)
        torch.set_deterministic(True)

        # Set requested number of threads
        torch.set_num_threads(options.num_thread)

        # Load training dataset
        with open('./dataset.pickle', 'rb') as input:
            data = pickle.load(input)

        self.train_set = data['train_set']

        # Create folders for output
        checkpoint_dir = './output_data/checkpoint'
        metadata_dir = './output_data/metadata'
        log_dir = './output_data/log'
        self.parameters_dir = './output_data/parameters'
        os.makedirs(checkpoint_dir, exist_ok = True)
        os.makedirs(metadata_dir, exist_ok = True)
        os.makedirs(log_dir, exist_ok = True)
        os.makedirs(self.parameters_dir, exist_ok = True)

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
        dt = 1.0 / 30.0
        self.motion_model = MotionModel(dt, 1.0).to(device = self.buddy.device)
        self.measurement_model = MeasurementModel(0.1) .to(device = self.buddy.device)
        # self.filter_model = FeatureExtendedKalmanFilter\
        self.filter_model = torchfilter.filters.ExtendedKalmanFilter\
        (
            dynamics_model = self.motion_model,
            measurement_model = self.measurement_model
        ).to(device = self.buddy.device)
        self.buddy.attach_model(self.filter_model)

        self.parameters = {}
        self.parameters['Q'] = []
        self.parameters['R'] = []


    def train(self, subsequence_length, epochs, batch_size = 32, initial_cov_scale = 0.1, recover_from = None):
        """Training procedure."""

        if recover_from is not None:
            self.buddy.load_checkpoint(label = recover_from)

        self.filter_model.train()

        dataloader = torch.utils.data.DataLoader\
        (
            torchfilter.data.SubsequenceDataset(trajectories = self.train_set, subsequence_length = subsequence_length),
            batch_size = batch_size,
            shuffle = True
        )

        initial_covariance = (torch.eye(self.filter_model.state_dim, device = self.buddy.device) * initial_cov_scale)

        for _ in range(epochs):
            torchfilter.train.train_filter\
            (
                self.buddy,
                self.filter_model,
                dataloader,
                initial_covariance = initial_covariance
            )
            # self.parameters['R'].append(self.filter_model.measurement_model.R.clone().detach().numpy())
            # self.parameters['Q'].append(self.filter_model.dynamics_model.Q.clone().detach().numpy())

        self.buddy.save_checkpoint('sub_' + str(subsequence_length))

        parameters_file = open(self.parameters_dir + '/parameters.pickle', 'wb')
        pickle.dump(self.parameters, parameters_file)
        parameters_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', dest = 'dataset_path', type = str, required = True)
    parser.add_argument('--number-thread', dest = 'num_thread', type = int, default = 2)

    options = parser.parse_args()

    trainer = Trainer(options)
    trainer.train(4, 5)
    trainer.train(8, 5)
    trainer.train(16, 5)
    # trainer.train(32, 5, recover_from = 'sub_16')
    # trainer.train(64, 5)
    # trainer.train(128, 5)

if __name__ == '__main__':
    main()
