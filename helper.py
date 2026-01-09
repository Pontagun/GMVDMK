import configparser


def get_gamma_filter(stilness, alpha):
    return (.5 * stilness) + (1 - .5) * alpha


def get_sensor_diff(v, w):
    config = configparser.ConfigParser()
    config.read('config.ini')
    mtnlns_threshold = float(config['SENSOR_THRESHOLD']['MTNLNSThreshold'])

    delta_vector = w - v
    delta_vector_max = max([abs(delta_vector.x)
                               , abs(delta_vector.y)
                               , abs(delta_vector.z)])

    if delta_vector_max <= mtnlns_threshold:
        return 1 - (delta_vector_max / mtnlns_threshold)
    else:
        return 0

