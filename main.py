import configparser
import pandas as pd
import statistics
import numpy as np

import helper
from sensor import QSensor
import matplotlib.pyplot as plt
from camera import Camera
import time
import quaternion


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    df = pd.read_csv(config['ENVIRONMENT']["Filename"])
    alpha_window = int(config['GMVDMK_WINDOW']['AlphaWindow'])

    df.columns = df.columns.str.replace(' ', '')
    data_row = df.shape[0]
    data_col = df.shape[1]

    q_G = np.quaternion(1, 0, 0, 0)
    q_G_lst = []
    temp = []

    camera = Camera(df)
    gyro = QSensor(df["gyro_x"], df["gyro_y"], df["gyro_z"])
    magnet = QSensor(df["mag_x"], df["mag_y"], df["mag_z"])
    accel = QSensor(df["acc_x"], df["acc_y"], df["acc_z"])

    m_init = magnet.quat[0]
    a_init = accel.quat[0]
    alpha_mtnlns = 0.0

    for i in range(alpha_window, data_row):

        delta_t = camera.get_delta_t(i) / 1000

        q_dot = .5 * (q_G * gyro.quat[i])
        power = delta_t * q_dot * q_G.conjugate()
        q_G = np.exp(power) * q_G
        q_G = QSensor.get_quat_normalization(q_G)

        v = accel.quat[i-alpha_window]
        w = accel.quat[i]
        stillness_acc = helper.get_sensor_diff(v, w)
        alpha_mtnlns = helper.get_gamma_filter(stillness_acc, alpha_mtnlns)

        q_G_lst.append(q_G)
        temp.append(alpha_mtnlns)

    plt.plot([val.x for val in q_G_lst])
    plt.plot([val.y for val in q_G_lst])
    plt.plot([val.z for val in q_G_lst])
    plt.plot([val.w for val in q_G_lst])
    plt.plot(temp)
    plt.show()
