#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import math
import numpy
import torch
import torch.nn as nn
import torchfilter
from fannypack.nn import resblocks


class MeasurementModel(torchfilter.base.KalmanFilterMeasurementModel):

    def __init__(self, initial_r):

        self.state_dim = 4
        self.measurement_dim = 4
        # self.measurement_dim = 2
        # self.inner_measurement_dim = 4

        super().__init__(state_dim = self.state_dim, observation_dim = self.measurement_dim)

        # Trainable parameters
        # self.R = torch.nn.Parameter(torch.cholesky(torch.diag(torch.FloatTensor([initial_r for i in range(self.measurement_dim)]))), requires_grad = True)
        units = 64
        self.state_feature = nn.Sequential(nn.Linear(self.state_dim, units), nn.ReLU(), resblocks.Linear(units, activation = 'relu'))
        self.R = nn.Sequential\
        (
            nn.Linear(units, units),
            resblocks.Linear(units, activation = 'relu'),
            resblocks.Linear(units, activation = 'relu'),
            resblocks.Linear(units, activation = 'relu'),
            nn.Linear(units, self.measurement_dim)
        )


    def forward(self, *, states):
        """Measurement prediction step."""

        N, state_dim = states.shape[:2]
        assert state_dim == self.state_dim

        # Sensors positions (batched)
        sensor_0 = torch.Tensor([0.0, 0.0])
        sensor_0 = sensor_0[None, :].repeat(N, 1)

        sensor_1 = torch.Tensor([0.0, 2.0])
        sensor_1 = sensor_1[None, :].repeat(N, 1)

        sensor_2 = torch.Tensor([2.0, 0.0])
        sensor_2 = sensor_2[None, :].repeat(N, 1)

        sensor_3 = torch.Tensor([2.0, 2.0])
        sensor_3 = sensor_3[None, :].repeat(N, 1)

        # Distance from predicted state to sensors
        # d_i = norm(state[0 : 2] - sensor_i)
        measurements = torch.zeros(N, self.measurement_dim).to(device = states.device)
        # measurements = torch.zeros(N, self.inner_measurement_dim).to(device = states.device)
        position = states[:, 0 : 2]
        measurements[:, 0] = torch.linalg.norm(position - sensor_0, dim = 1)
        measurements[:, 1] = torch.linalg.norm(position - sensor_1, dim = 1)
        measurements[:, 2] = torch.linalg.norm(position - sensor_2, dim = 1)
        measurements[:, 3] = torch.linalg.norm(position - sensor_3, dim = 1)
        # measurements = self.predicted_feature(states / 2.0)

        Rs = torch.diag_embed(torch.exp(self.R(self.state_feature(states / 2.0))) + torch.Tensor([.001] * self.measurement_dim))
        # Rs = torch.exp(self.R(self.state_feature(states / 2.0)).reshape(N, self.measurement_dim, self.measurement_dim))
        # Rs = self.R[None, :, :].expand(N, self.measurement_dim, self.measurement_dim)
        test = Rs @ Rs.transpose(-1, -2)
        test_chol = torch.linalg.cholesky(test)

        return measurements, Rs
