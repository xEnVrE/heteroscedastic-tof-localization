import numpy as np
from typing import NamedTuple

def split_trajectories(trajectories, length):

    subsequences = []

    for traj in trajectories:
        assert traj.states.shape[0] == traj.inputs.shape[0]
        assert traj.states.shape[0] == traj.measurements.shape[0]

        for offset in (0, length // 2):

            def split_x(x):
                x = x[offset:]

                number_sections = x.shape[0] // length
                new_length = number_sections * length
                x = x[:new_length]

                return np.split(x, number_sections)

            for x, y, u in zip(split_x(traj.states), split_x(traj.measurements), split_x(traj.inputs)):
                subsequences.append(Trajectory(states = x, measurements = y, inputs = u))

    return subsequences


class Trajectory(NamedTuple):

    states: np.ndarray
    measurements: np.ndarray
    inputs: np.ndarray


def main():
    x0 = np.array(range(1, 7))
    y0 = np.array(range(7, 13))
    u0 = np.array(range(13, 19))

    x1 = np.array(range(19, 25))
    y1 = np.array(range(25, 31))
    u1 = np.array(range(31, 37))

    t0 = Trajectory(states = x0, inputs = u0, measurements = y0)
    t1 = Trajectory(states = x1, inputs = u1, measurements = y1)

    trajectories = []
    trajectories.append(t0)
    trajectories.append(t1)

    print(t0)
    print(t1)
    t = split_trajectories(trajectories, length = 2)
    print(len(t))
    print(t)

if __name__ == '__main__':
    main()
