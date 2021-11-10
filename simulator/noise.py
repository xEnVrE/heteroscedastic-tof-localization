#===============================================================================
#
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# GPL-2+ license. See the accompanying LICENSE file for details.
#
#===============================================================================

import abc
import numpy


class Noise(abc.ABC):

    def __init__(self):
        """Constructor"""

        numpy.random.seed(0)


    @abc.abstractmethod
    def sample(self, *, sensor_position, robot_position):
        """Noise sampler."""
