#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import numpy
from overrides import overrides
from simulator.noise import Noise


class VaryingNoise(Noise):

    def __init__(self, distance_base = 0.5, variance_base = 1.0, variance_change = 1.0):
        """Constructor."""

        super().__init__()

        self.distance_base = distance_base
        self.variance_base = variance_base
        self.variance_change = variance_change


    @overrides
    def sample(self, *, sensor_position, robot_position):
        """Implement normal noise with
        variance = variance_ + growth_gain * (distance_base - norm(sensor - robot)) ** 2
        """

        d = numpy.linalg.norm(sensor_position - robot_position)
        variance = self.variance_base #+ self.growth_gain * (self.distance_base - d) ** 2

        if robot_position[1] > robot_position[0]:
            if robot_position[1] < robot_position[0] + 1.0:
                variance = self.variance_change

        # Note: scale here is the std actually, but we use it as the variance
        return numpy.random.normal(scale = variance)


def main():

    noise = VaryingNoise(0.5, 0.01, 0.07)
    sensor_position = numpy.zeros(2)
    robot_positions = [numpy.array([0.0, d]) for d in numpy.linspace(0.0, 2.0, 5)]

    number_samples = 100
    noise_samples = []
    for robot_position in robot_positions:
        noise_samples.append([noise.sample(sensor_position = sensor_position, robot_position = robot_position) for i in range(number_samples)])

    import matplotlib.pyplot as plt
    for i, robot_position in enumerate(robot_positions):
        plt.plot(noise_samples[i])

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
