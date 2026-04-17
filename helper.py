import configparser

import quaternion


def get_gamma_filter(stilness, alpha):
    return (.25 * stilness) + (1 - .25) * alpha


def get_sensor_diff(hist_accel, curr_accel):
    config = configparser.ConfigParser()
    config.read('config.ini')
    mtnlns_threshold = float(config['SENSOR_THRESHOLD']['MTNLNSThreshold'])

    delta_vector = curr_accel - hist_accel
    delta_vector_max = max([abs(delta_vector.x)
                               , abs(delta_vector.y)
                               , abs(delta_vector.z)])

    if delta_vector_max <= mtnlns_threshold:
        return 1 - (delta_vector_max / .25)
    else:
        return 0


def get_vector(q):
    if type(q) == quaternion.quaternion:
        return quaternion.as_vector_part(q)
    return None
