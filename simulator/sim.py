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
from simulator.sensor import Sensor
from simulator.varying_noise import VaryingNoise
from simulator.viewer import Viewer


class Simulator():

    def __init__(self, dt = 1.0 / 30.0, psd = 0.1, limit = 2.0):
        """Constructor."""

        numpy.random.seed(0)

        # Sampling time and total time
        self.dt = dt
        self.t = 0

        # Space for the robot limited to [0, limit] x [0, limit] <= R2
        self.limit = limit
        self.barriers_gains = [0.1] * 4

        # Saturation for the velocity of the robot
        self.max_velocity = 1.0

        # Covariance modelling white noise acceleration (WNA) motion model
        self.covariance = numpy.array\
        ([
            [self.dt ** 3 / 3, 0.0,              self.dt ** 2 / 2, 0.0],
            [0.0,              self.dt ** 3 / 2, 0.0,              self.dt ** 2 / 2],
            [self.dt ** 2 / 2, 0.0,              self.dt,          0.0],
            [0.0,              self.dt ** 2 / 2, 0.0,              self.dt]
        ]) * psd

        # State for the robot initialized with zero velocities and
        # initial position in (limit, limit) / 2
        self.state = numpy.zeros(4)

        # Initialize tof sensors
        # self.sensors = [Sensor(i, j, VaryingNoise(0.0, 0.01, 0.015)) for i in [0.0, self.limit] for j in [0.0, self.limit]]
        # self.sensors = [Sensor(i, j, VaryingNoise(0.0, 0.025, 0.0 * 0.015)) for i in [0.0, self.limit] for j in [0.0, self.limit]]
        self.sensors = [Sensor(i, j, VaryingNoise(0.0, 0.01, 0.07)) for i in [0.0, self.limit] for j in [0.0, self.limit]]
        self.sensors_gt = [Sensor(i, j, VaryingNoise(0.0, 0.0, 0.0)) for i in [0.0, self.limit] for j in [0.0, self.limit]]
        print("Sensor positions are:")
        for i in range(len(self.sensors)):
            print(self.sensors[i].sensor_position())

        self.reset()


    def reset(self):
        """Reset the simulator."""

        # Initialize storage
        self.history_time = []
        self.history_state = []
        self.history_sensors = {i : [] for i in range(len(self.sensors))}
        self.history_sensors_gt = {i : [] for i in range(len(self.sensors))}

        # Set initial condition
        self.state[0 : 2] = numpy.ones(2) * self.limit / 2


    def robot_state(self):
        """Return the current robot state."""

        return numpy.copy(self.state)


    def robot_position(self):
        """Return the current robot position."""

        return numpy.copy(self.state[0 : 2])


    def robot_history(self):
        """Return the entire robot state history."""

        return numpy.array(self.history_state, dtype = numpy.float32)


    def sensors_history(self, ground_truth = False):
        """Return the entire sensors history."""

        if ground_truth:
            return numpy.array([numpy.array(self.history_sensors_gt[i], dtype = numpy.float32) for i in range(len(self.history_sensors_gt))], dtype = numpy.float32).transpose()

        return numpy.array([numpy.array(self.history_sensors[i], dtype = numpy.float32) for i in range(len(self.history_sensors))], dtype = numpy.float32).transpose()


    def sensors_positions(self):
        """Return the sensors positions."""

        return numpy.array([self.sensors[i].sensor_position() for i in range(len(self.sensors))], dtype = numpy.float32)


    def time_history(self):
        """Return the entire time history."""

        return numpy.copy(numpy.array(self.history_time))


    def artificial_repulsive_field(self, distance, clearance = 0.1, gamma = 2, gain = 0.01):
        """Implement a basic repulsive potential field of the form
             __
            |
            |  k / gamma (1 / distance - 1 / clearance) ** gamma, distance <= clearance
        u = |
            |  0                                                , distance > clearance
            |__
        """

        if distance <= clearance:
            return gain / gamma * (1 / distance - 1 / clearance) ** gamma

        return 0


    def barriers(self, state):
        """Evaluate control input that moves the robot away from the barriers."""

        ctl = numpy.zeros(2)
        distances = []
        for i in range(2):
            distances.append(abs(state[i]))
            distances.append(abs(state[i] - self.limit))

        barriers = [self.artificial_repulsive_field(d) for d in distances]

        for i in range(2):
            ctl[i] = (barriers[2 * i] - barriers[2 * i + 1])

        return ctl


    def step(self):
        """Step the simulator."""

        # Sample noise
        noise = numpy.random.multivariate_normal(mean = numpy.zeros(4), cov = self.covariance)

        # Step white noise acceleration model
        state = self.state + noise
        state[0 : 2] += self.state[2 :] * self.dt

        # Saturate robot velocities
        for i in range(2):
            if abs(state[2 + i]) > self.max_velocity:
                state[2 + i] = math.copysign(self.max_velocity, state[2 + i])

        # Add respulsive control to avoid barriers
        ctl = self.barriers(state)
        state[2 : ] += ctl

        self.state = state

        # Append to internal history
        self.history_state.append(self.robot_state())
        for j, sensor in enumerate(self.sensors):
            self.history_sensors[j].append(sensor.distance(self.robot_position()))
        for j, sensor in enumerate(self.sensors_gt):
            self.history_sensors_gt[j].append(sensor.distance(self.robot_position()))
        self.history_time.append(self.t)
        self.t += self.dt


def main():

    # Setup simulator
    dt = 1.0 / 30.0
    psd = 0.1
    limit = 2.0
    s = Simulator(dt = dt, psd = psd, limit = limit)

    # Simulate
    seconds = 20.0
    s.reset()
    for i in range(int(seconds / dt)):
        s.step()

    # Show
    v = Viewer(limit, s.sensors_positions())
    v.show(s.time_history(), s.robot_history(), s.sensors_history(), s.sensors_history(ground_truth = True))

if __name__ == '__main__':
    main()
