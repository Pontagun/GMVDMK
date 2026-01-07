import configparser
import pandas as pd
import statistics
import numpy as np
from sensor import QSensor
import matplotlib.pyplot as plt
from camera import Camera
import time
import quaternion


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    df = pd.read_csv("REC_01-05/rec01.csv")
    df.columns = df.columns.str.replace(' ', '')

    data_row = df.shape[0]
    data_col = df.shape[1]

    q_G = np.quaternion(1, 0, 0, 0)

    raw_q_G = []

    q_zero = np.quaternion(1, 0, 0, 0)

    # List of each column.
    camera = Camera(df)
    temp_for_plot = []
    # List of quaternion.
    gyro = QSensor([df["gyro_x"], df["gyro_y"], df["gyro_z"]])
    magnet = QSensor([df["mag_x"], df["mag_y"], df["mag_z"]])
    accel = QSensor([df["acc_x"], df["acc_y"], df["acc_z"]])

    for i in range(1, data_row):

        q_dot = .5 * (q_zero * gyro.quat[i])

        delta_t = camera.get_delta_t(i) / 1000
        power = delta_t * q_dot * q_G.conjugate()
        q_G = np.exp(power) * q_zero

        q_G = QSensor.get_quat_normalization(q_G)
        q_zero = q_G


        raw_q_G.append(q_G)

    plt.plot([val.x for val in raw_q_G])
    plt.plot([val.y for val in raw_q_G])
    plt.plot([val.z for val in raw_q_G])
    plt.plot([val.w for val in raw_q_G])
    # plt.plot(temp_for_plot)
    plt.show()
