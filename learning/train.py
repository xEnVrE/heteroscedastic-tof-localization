#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import argparse
import pickle
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from filter.motion_model import MotionModel
from filter.measurement_model import MeasurementModel
from filter.ekf import ExtendedKalmanFilter
from learning.dataset import DatasetOfSubsequences
from tqdm import tqdm

class Trainer():

    def __init__(self, options):
        """Constructor."""

        self.device = torch.device('cpu')
        self.batch_size = 32
        torch.manual_seed(0)

        # Set requested number of threads
        torch.set_num_threads(options.num_thread)

        # Load training dataset
        with open('./dataset.pickle', 'rb') as input:
            data = pickle.load(input)

        self.train_set = data['train_set']

        # Create folders for output
        checkpoint_dir = './output_data/checkpoint'
        os.makedirs(checkpoint_dir, exist_ok = True)

        # Filter initialization
        dt = 1.0 / 30.0
        self.motion_model = MotionModel(dt, 1.0, self.device).to(device = self.device)
        self.measurement_model = MeasurementModel(0.1, self.batch_size, self.device).to(device = self.device)
        self.filter_model = ExtendedKalmanFilter\
        (
            motion_model = self.motion_model,
            measurement_model = self.measurement_model
        ).to(device = self.device)

        # Storage for output
        self.output_dir = './train_' + datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        os.makedirs(self.output_dir, exist_ok = False)


    def train(self, subsequence_length, epochs, initial_cov_scale = 0.1, recover_from = None):
        """Training procedure."""

        # if recover_from is not None:
        #     self.buddy.load_checkpoint(label = recover_from)

        self.loss_history = []
        self.filter_model.train()
        self.optimizer = torch.optim.Adam(self.filter_model.parameters())

        dataloader = torch.utils.data.DataLoader\
        (
            DatasetOfSubsequences(trajectories = self.train_set, length = subsequence_length),
            batch_size = self.batch_size,
            shuffle = True
        )

        initial_covariance = (torch.eye(self.filter_model.state_dim) * initial_cov_scale).to(self.device)

        for epoch_number in range(epochs):
            assert initial_covariance.shape == (self.filter_model.state_dim, self.filter_model.state_dim)
            assert self.filter_model.training, "Model needs to be set to train mode"

            epoch_loss = 0.0

            for batch_idx, batch_data in enumerate(tqdm(dataloader)):
                states = batch_data.states.to(self.device)
                measurements = batch_data.measurements.to(self.device)
                inputs = batch_data.inputs.to(self.device)

                # as all the code was designed to handle (T, B, *) tensors
                states = torch.transpose(states, 0, 1)
                measurements = torch.transpose(measurements, 0, 1)
                inputs = torch.transpose(inputs, 0, 1)

                T, B, state_dim = states.shape
                assert state_dim == self.filter_model.state_dim
                assert measurements.shape[:2] == (T, B)
                assert inputs.shape[:2] == (T, B)
                assert batch_idx != 0 or B == dataloader.batch_size

                initial_states_covariance = initial_covariance[None, :, :].expand\
                ((B, state_dim, state_dim))

                initial_states = torch.distributions.MultivariateNormal\
                (
                    states[0], covariance_matrix = initial_states_covariance
                ).sample()

                self.filter_model.initialize_beliefs(mean = initial_states, covariance = initial_states_covariance)

                state_predictions = self.filter_model.forward_loop\
                (
                    measurements = measurements[1:],
                    inputs = inputs[1:]
                )
                assert state_predictions.shape == (T - 1, B, state_dim)

                loss = F.mse_loss(state_predictions, states[1:])
                self.optimizer.zero_grad()
                loss.backward(retain_graph = False)
                self.optimizer.step()

                epoch_loss += loss.detach().cpu().numpy()


            epoch_loss /= len(dataloader)
            self.loss_history.append(epoch_loss)
            self.save_epoch(epoch_number, subsequence_length)
            print('(train_filter) Epoch training loss: ', epoch_loss)


    def save_epoch(self, epoch_number, subsequence_length):
        """Save the current epoch."""

        output_dictionary = {}
        output_dictionary['epoch_number'] = epoch_number
        output_dictionary['subsequence_length'] = subsequence_length
        output_dictionary['output_dir'] = self.output_dir
        output_dictionary['filter'] = self.filter_model.state_dict()
        output_dictionary['optimizer'] = self.optimizer.state_dict()
        output_dictionary['loss_history'] = self.loss_history

        torch.save(output_dictionary, self.output_dir + '/' + str(subsequence_length) + '_' + str(epoch_number).zfill(4) + '.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', dest = 'dataset_path', type = str, required = True)
    parser.add_argument('--number-thread', dest = 'num_thread', type = int, default = 2)

    options = parser.parse_args()

    trainer = Trainer(options)
    trainer.train(4, 5)
    trainer.train(8, 5)
    trainer.train(16, 5)


if __name__ == '__main__':
    main()
