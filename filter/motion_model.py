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
from fannypack.nn import resblocks


class MotionModel(nn.Module):

    def __init__(self, sampling_time, initial_q, device):
        """Constructor."""
        super().__init__()

        self.state_dim = 4

        # Fixed parameters
        self.dt = nn.Parameter(torch.FloatTensor([sampling_time]), requires_grad = False)
        self.eps = torch.Tensor([.001] * self.state_dim).to(device)

        # Trainable parameters
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
        self.zeros = torch.zeros(self.state_dim).to(device)


    def forward(self, *, initial_states, inputs):
        """Prediction step."""

        B, state_dim = initial_states.shape[:2]
        assert state_dim == self.state_dim

        # Constant velocity model
        # x_k = x_k-1 + v_k-1 * dt
        # v_k = v_k-1
        predicted_states = self.zeros[None, :].repeat(B, 1)
        predicted_states[:, 0 : 2] = initial_states[:, 0 : 2] + initial_states[:, 2 :] * self.dt
        predicted_states[:, 2 :] = initial_states[:, 2 :]

        Qsqrt = torch.diag_embed(torch.exp(self.Q(self.state_feature(initial_states / 2.0))) + self.eps)
        return predicted_states, Qsqrt


    def forward_loop(self, *, initial_states, inputs):
        """Prediction step over a sequence of given length T and for a given batch B."""

        T, B = inputs.shape[:2]

        assert initial_states.shape == (B, self.state_dim)
        assert T > 0

        x_list = []
        Qsqrt_list = []

        x = initial_states
        for k in range(T):
            x, Qsqrt = self(initial_states = x, inputs = inputs[k])

            assert x.shape == (B, self.state_dim)
            assert Qs.shape == (B, self.state_dim, self.state_dim)

            x_list.append(x)
            Qsqrt_list.append(Qsqrt)

        x_tensor = torch.stack(x_list, dim = 0)
        Qsqrt_tensor = torch.stack(Qsqrt_list, dim = 0)

        assert x_tensor.shape == (T, B, self.state_dim)
        assert Qsqrt_tensor.shape == (T, B, self.state_dim, self._dim)

        return x_tensor, Qsqrt_tensor


    def jacobian(self, initial_states, inputs):
        """Partial derivative of model w.r.t. the state evaluated in initial_states."""

        with torch.enable_grad():
            x = initial_states.detach().clone()

            B, state_dim = x.shape
            assert state_dim == self.state_dim

            # insert additional dimension between B and state_dim and copy the state state_dim time
            # the result will have shape (B, state_dim, state_dim)
            x = x[:, None, :].expand((B, state_dim, state_dim))

            # prepare the inputs for all the repetions of the state as above
            # the result will have shape (B * state_dim, state_dim)
            inputs = torch.repeat_interleave(inputs, repeats=state_dim, dim=0)

            x.requires_grad_(True)

            # evaluate output
            # reshape is required as the forward method assumes an input of size (batch, state dim)
            # second reshape is required as we need the same size as x
            y = self(initial_states = x.reshape((-1, state_dim)), inputs = inputs)[0].reshape(B, state_dim, state_dim)

            mask = torch.eye(state_dim, device = x.device).repeat(B, 1, 1)
            jac = torch.autograd.grad(y, x, mask, create_graph = True)

        return jac[0]
