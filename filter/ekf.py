import math
import torch
from torch import nn
from typing import cast


class ExtendedKalmanFilter(nn.Module):
    """Generic differentiable EKF."""

    def __init__(self, motion_model, measurement_model):
        super().__init__()
        self.state_dim = motion_model.state_dim
        self.motion_model = motion_model
        self.measurement_model = measurement_model

        self._belief_mean: torch.Tensor
        self._belief_covariance: torch.Tensor

        self._initialized = False


    def initialize_beliefs(self, *, mean, covariance):

        B = mean.shape[0]
        assert mean.shape == (B, self.state_dim)
        assert covariance.shape == (B, self.state_dim, self.state_dim)

        self.belief_mean = mean
        self.belief_covariance = covariance

        self._initialized = True


    @property
    def belief_mean(self):
        return self._belief_mean


    @belief_mean.setter
    def belief_mean(self, mean):
        self._belief_mean = mean


    @property
    def belief_covariance(self):
        return self._belief_covariance


    @belief_covariance.setter
    def belief_covariance(self, covariance):
        self._belief_covariance = covariance


    def forward(self, *, measurements, inputs):

        assert self._initialized

        B, state_dim = self.belief_mean.shape
        assert measurements.shape[0] == B
        assert inputs.shape[0] == B

        self._predict_step(inputs = inputs)
        self._update_step(measurements = measurements)

        return self.belief_mean


    def forward_loop(self, *, measurements, inputs):

        T, B = inputs.shape[:2]

        assert measurements.shape[:2] == (T, B)

        k = 0
        x = self(measurements = measurements[k], inputs = inputs[k])
        assert x.shape == (B, self.state_dim)

        x_tensor = x.new_zeros((T, B, self.state_dim))
        x_tensor[k] = x

        for k in range(1, T):
            x = self(measurements = measurements[k], inputs = inputs[k])

            assert x.shape == (B, self.state_dim)

            x_tensor[k] = x

        assert x_tensor.shape == (T, B, self.state_dim)

        return x_tensor


    def _predict_step(self, *, inputs):
        # Get previous belief
        prev_mean = self._belief_mean
        prev_covariance = self._belief_covariance
        B, state_dim = prev_mean.shape

        # Compute mu_{t+1|t}, covariance, and Jacobian
        pred_mean, dynamics_tril = self.motion_model(
            initial_states=prev_mean, inputs=inputs
        )
        dynamics_covariance = dynamics_tril @ dynamics_tril.transpose(-1, -2)
        dynamics_A_matrix = self.motion_model.jacobian(
            initial_states=prev_mean, inputs=inputs
        )
        assert dynamics_covariance.shape == (B, state_dim, state_dim)
        assert dynamics_A_matrix.shape == (B, state_dim, state_dim)

        # Calculate Sigma_{t+1|t}
        pred_covariance = (
            dynamics_A_matrix @ prev_covariance @ dynamics_A_matrix.transpose(-1, -2)
            + dynamics_covariance
        )

        # Update internal state
        self._belief_mean = pred_mean
        self._belief_covariance = pred_covariance


    def _update_step(self, *, measurements):
        # Extract/validate inputs

        # assert isinstance(
        #     measurements, types.MeasurementsNoDictTorch
        # ), "For standard EKF, measurements must be tensor!"
        # measurements = cast(types.MeasurementsNoDictTorch, measurements)

        pred_mean = self._belief_mean
        pred_covariance = self._belief_covariance

        # Measurement model forward pass, Jacobian
        measurements_mean = measurements
        pred_measurements, measurements_tril = self.measurement_model(states=pred_mean)
        measurements_covariance = measurements_tril @ measurements_tril.transpose(
            -1, -2
        )
        C_matrix = self.measurement_model.jacobian(states=pred_mean)
        assert measurements_mean.shape == pred_measurements.shape

        # Check shapes
        B, observation_dim = measurements_mean.shape
        assert measurements_covariance.shape == (B, observation_dim, observation_dim)
        assert measurements_mean.shape == (B, observation_dim)

        # Compute Kalman Gain, innovation
        innovation = measurements_mean - pred_measurements
        innovation_covariance = (
            C_matrix @ pred_covariance @ C_matrix.transpose(-1, -2)
            + measurements_covariance
        )
        kalman_gain = (
            pred_covariance
            @ C_matrix.transpose(-1, -2)
            @ torch.inverse(innovation_covariance)
        )

        # Get mu_{t+1|t+1}, Sigma_{t+1|t+1}
        corrected_mean = pred_mean + (kalman_gain @ innovation[:, :, None]).squeeze(-1)
        assert corrected_mean.shape == (B, self.state_dim)

        identity = torch.eye(self.state_dim, device=kalman_gain.device)
        corrected_covariance = (identity - kalman_gain @ C_matrix) @ pred_covariance
        assert corrected_covariance.shape == (B, self.state_dim, self.state_dim)

        # Update internal state
        self._belief_mean = corrected_mean
        self._belief_covariance = corrected_covariance
