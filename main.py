import configparser
import pandas as pd
import statistics
import numpy as np
from sensor import quaternion_sensor
import matplotlib.pyplot as plt
from camera import camera_info
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
    KM = np.zeros(data_row)

    start = time.time()

    # List of each column.
    camera = camera_info(df)
    delta_t = []
    temp_for_plot = []
    # List of quaternion.
    gyro = quaternion_sensor([df["gyro_x"], df["gyro_y"], df["gyro_z"]])
    magnet = quaternion_sensor([df["mag_x"], df["mag_y"], df["mag_z"]])
    accel = quaternion_sensor([df["acc_x"], df["acc_y"], df["acc_z"]])

    for i in range(1, data_row):
        temp_for_plot.append(np.linalg.norm(quaternion.as_float_array(gyro.quaternion_signal[i])))

        q_dot = .5 * (q_zero * gyro.quaternion_signal[i])

        delta_t = camera.get_delta_t(i) / 1000
        power_of_e = delta_t * q_dot * q_G.conjugate()
        q_G = np.exp(power_of_e) * q_zero
        q_zero = q_G
        raw_q_G.append(q_G)
    print(time.time() - start)

    plt.plot([val.x for val in raw_q_G])
    plt.plot([val.y for val in raw_q_G])
    plt.plot([val.z for val in raw_q_G])
    # plt.plot([val.w for val in raw_q_G])
    # plt.plot(temp_for_plot)
    plt.show()
