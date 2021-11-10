#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import numpy
import torch
import torch.nn as nn
import torchfilter
from fannypack.nn import resblocks


class MotionModel(torchfilter.base.DynamicsModel):

    def __init__(self, sampling_time, initial_q):
        """Constructor."""

        self.state_dim = 4

        super().__init__(state_dim = self.state_dim)

        # Fixed parameters
        self.dt = nn.Parameter(torch.FloatTensor([sampling_time]), requires_grad = False)

        # Trainable parameters
        # self.Q = nn.Parameter(torch.cholesky(torch.diag(torch.FloatTensor([initial_q for i in range(self.state_dim)]))), requires_grad = True)
        units = 64
        self.state_feature = nn.Sequential(nn.Linear(self.state_dim, units), nn.ReLU(), resblocks.Linear(units, activation = 'relu'))
        self.Q = nn.Sequential\
        (
            nn.Linear(units, units),
            resblocks.Linear(units, activation = 'relu'),
            resblocks.Linear(units, activation = 'relu'),
            resblocks.Linear(units, activation = 'relu'),
            nn.Linear(units, self.state_dim)
        )


    def forward(self, *, initial_states, controls):
        """Prediction step."""

        N, state_dim = initial_states.shape[:2]
        assert state_dim == self.state_dim

        # Constant velocity model
        # x_k = x_k-1 + v_k-1 * dt
        # v_k = v_k-1
        predicted_states = torch.zeros(initial_states.size()).to(device = initial_states.device)
        predicted_states[:, 0 : 2] = initial_states[:, 0 : 2] + initial_states[:, 2 :] * self.dt
        predicted_states[:, 2 :] = initial_states[:, 2 :]

        # Qs = self.Q[None, :, :].expand(N, state_dim, state_dim)
        Qs = torch.diag_embed(torch.exp(self.Q(self.state_feature(initial_states / 2.0))) + torch.Tensor([.001] * self.state_dim))
        test = Qs @ Qs.transpose(-1, -2)
        test_chol = torch.cholesky(test)
        # Qs = self.Q(self.state_feature(initial_states / 2.0)).reshape((N, self.state_dim, self.state_dim))
        return predicted_states, Qs
