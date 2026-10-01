import configparser

import numpy as np
import quaternion

config = configparser.ConfigParser()
config.read('config.ini')
mtnlns_threshold = float(config['SENSOR_THRESHOLD']['MTNLNSThreshold'])


def get_gamma_filter(stillness, alpha_init=1.0):
    # Exponential smoothing, alpha[n] = .25 * stillness[n] + .75 * alpha[n - 1], starting from alpha_init.
    alpha = []
    prev = alpha_init
    for value in stillness.tolist():
        prev = (.25 * value) + (1 - .25) * prev
        alpha.append(prev)

    return alpha


def get_sensor_diff(accel_quat, window):
    # Stillness of sample i from the accel change against sample i - window, for every i >= window.
    accel = quaternion.as_vector_part(accel_quat)
    delta_vector_max = np.abs(accel[window:] - accel[:-window]).max(axis=1)

    return np.where(delta_vector_max <= mtnlns_threshold, 1 - (delta_vector_max / .25), 0)


def get_vector(q):
    if type(q) == quaternion.quaternion:
        return quaternion.as_vector_part(q)
    return None


def get_quat_moving_average(q_lst, window):
    # Trailing moving average (current sample included); the first samples average over what is available.
    q = quaternion.as_float_array(np.asarray(q_lst))
    smoothed = np.empty_like(q)

    for i in range(len(q)):
        q_window = q[max(0, i - window + 1):i + 1]
        # q and -q are the same rotation, so flip samples onto the current sample's hemisphere before averaging.
        q_window = q_window * np.where(q_window @ q[i] < 0, -1, 1)[:, None]
        mean = q_window.mean(axis=0)
        smoothed[i] = mean / np.linalg.norm(mean)

    return list(quaternion.as_quat_array(smoothed))
