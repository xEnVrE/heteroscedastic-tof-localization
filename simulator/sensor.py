#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import numpy
from simulator.noise import Noise


class Sensor():

    def __init__(self, x, y, noise = None):
        """Constructor:"""

        self.x = x
        self.y = y
        self.noise = noise


    def sensor_position(self):
        """Return the absolute position of the sensor."""

        return numpy.array([self.x, self.y])


    def distance(self, robot_position):
        """Return the distance of the sensor from the robot."""

        d = numpy.linalg.norm(self.sensor_position() - robot_position)

        if self.noise is not None:
            d += self.noise.sample(sensor_position = self.sensor_position(), robot_position = robot_position)

        return d


def main():

    import mock

    constant_noise = mock
    constant_noise.sample = lambda **kwargs : 0.5

    s = Sensor(1.0, 1.0, constant_noise)

    robot_position = numpy.array([1.0, 2.0])

    print(s.distance(robot_position))

if __name__ == '__main__':
    main()
