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
from fannypack.nn import resblocks


class MeasurementModel(nn.Module):

    def __init__(self, initial_r, batch_size, device):
        super().__init__()

        self.state_dim = 4
        self.measurement_dim = 4
        self.eps = torch.Tensor([.001] * self.measurement_dim).to(device)
        self.batch_size = batch_size

        # Trainable parameters
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

        self.sensor_0 = torch.Tensor([0.0, 0.0]).to(device)
        self.sensor_1 = torch.Tensor([0.0, 2.0]).to(device)
        self.sensor_2 = torch.Tensor([2.0, 0.0]).to(device)
        self.sensor_3 = torch.Tensor([2.0, 2.0]).to(device)
        self.zeros. = torch.zeros(self.measurement_dim).to(device)


    def forward(self, *, states):
        """Measurement prediction step."""

        B, state_dim = states.shape[:2]
        assert state_dim == self.state_dim

        # Distance from predicted state to sensors
        measurements = self.zeros[None, :].repeat(B, 1)
        position = states[:, 0 : 2]

        # Sensors positions (batched)
        sensor_0 = self.sensor_0[None, :].repeat(B, 1)
        sensor_1 = self.sensor_1[None, :].repeat(B, 1)
        sensor_2 = self.sensor_2[None, :].repeat(B, 1)
        sensor_3 = self.sensor_3[None, :].repeat(B, 1)

        measurements[:, 0] = torch.linalg.norm(position - sensor_0, dim = 1)
        measurements[:, 1] = torch.linalg.norm(position - sensor_1, dim = 1)
        measurements[:, 2] = torch.linalg.norm(position - sensor_2, dim = 1)
        measurements[:, 3] = torch.linalg.norm(position - sensor_3, dim = 1)

        Rs = torch.diag_embed(torch.exp(self.R(self.state_feature(states / 2.0))) + self.eps)

        return measurements, Rs


    def jacobian(self, states):
        """Partial derivative of model w.r.t. the state evaluated in states."""

        measurement_dim = self.measurement_dim

        with torch.enable_grad():
            x = states.detach().clone()

            B, state_dim = x.shape
            assert state_dim == self.state_dim

            x = x[:, None, :].expand((B, measurement_dim, state_dim))
            x.requires_grad_(True)

            y = self(states = x.reshape((-1, state_dim)))[0].reshape((B, -1, measurement_dim))
            mask = torch.eye(measurement_dim, device=x.device).repeat(B, 1, 1)
            jac = torch.autograd.grad(y, x, mask, create_graph=True)

        return jac[0]
