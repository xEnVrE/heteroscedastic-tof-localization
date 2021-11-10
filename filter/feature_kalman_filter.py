import math
import torch
from overrides import overrides
from torch import nn
from torchfilter import types
from torchfilter.base import KalmanFilterBase
from typing import cast


class FeatureExtendedKalmanFilter(KalmanFilterBase):
    """Generic differentiable FEKF.

    """

    def __init__(self, dynamics_model, measurement_model):
        """
        Constructor.
        """

        super().__init__(dynamics_model = dynamics_model, measurement_model = measurement_model)


    @overrides
    def _predict_step(self, *, controls: types.ControlsTorch) -> None:
        # Get previous belief
        prev_mean = self._belief_mean
        prev_covariance = self._belief_covariance
        N, state_dim = prev_mean.shape

        # Compute mu_{t+1|t}, covariance, and Jacobian
        pred_mean, dynamics_tril = self.dynamics_model(
            initial_states=prev_mean, controls=controls
        )
        dynamics_covariance = dynamics_tril @ dynamics_tril.transpose(-1, -2)
        dynamics_A_matrix = self.dynamics_model.jacobian(
            initial_states=prev_mean, controls=controls
        )
        assert dynamics_covariance.shape == (N, state_dim, state_dim)
        assert dynamics_A_matrix.shape == (N, state_dim, state_dim)

        # Calculate Sigma_{t+1|t}
        pred_covariance = (
            dynamics_A_matrix @ prev_covariance @ dynamics_A_matrix.transpose(-1, -2)
            + dynamics_covariance
        )

        # Update internal state
        self._belief_mean = pred_mean
        self._belief_covariance = pred_covariance


    @overrides
    def _update_step(self, *, observations: types.ObservationsTorch) -> None:
        # Extract/validate inputs
        assert isinstance(
            observations, types.ObservationsNoDictTorch
        ), "For standard EKF, observations must be tensor!"
        observations = cast(types.ObservationsNoDictTorch, observations)
        pred_mean = self._belief_mean
        pred_covariance = self._belief_covariance

        # Measurement model forward pass, Jacobian
        observations_mean = self.measurement_model.measurement_feature(observations / (2 * math.sqrt(2.0)))
        pred_observations, observations_tril = self.measurement_model(states=pred_mean)
        observations_covariance = observations_tril @ observations_tril.transpose(
            -1, -2
        )
        C_matrix = self.measurement_model.jacobian(states=pred_mean)
        assert observations_mean.shape == pred_observations.shape

        # Check shapes
        N, observation_dim = observations_mean.shape
        assert observations_covariance.shape == (N, observation_dim, observation_dim)
        assert observations_mean.shape == (N, observation_dim)

        # Compute Kalman Gain, innovation
        innovation = observations_mean - pred_observations
        # innovation = self.features(torch.cat((pred_mean, observations_mean), dim=-1)) - self.features(torch.cat((pred_mean, pred_observations), dim=-1))
        innovation_covariance = (
            C_matrix @ pred_covariance @ C_matrix.transpose(-1, -2)
            + observations_covariance
        )
        kalman_gain = (
            pred_covariance
            @ C_matrix.transpose(-1, -2)
            @ torch.inverse(innovation_covariance)
        )

        # Get mu_{t+1|t+1}, Sigma_{t+1|t+1}
        corrected_mean = pred_mean + (kalman_gain @ innovation[:, :, None]).squeeze(-1)
        assert corrected_mean.shape == (N, self.state_dim)

        identity = torch.eye(self.state_dim, device=kalman_gain.device)
        corrected_covariance = (identity - kalman_gain @ C_matrix) @ pred_covariance
        assert corrected_covariance.shape == (N, self.state_dim, self.state_dim)

        # Update internal state
        self._belief_mean = corrected_mean
        self._belief_covariance = corrected_covariance
