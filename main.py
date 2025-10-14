import configparser
import pandas as pd
import statistics
import numpy as np
from sensor import quaternion_sensor
import matplotlib.pyplot as plt
from camera import camera_info
import time

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    df = pd.read_csv("REC_01-05/rec01.csv")
    df.columns = df.columns.str.replace(' ', '')

    data_row = df.shape[0]
    data_col = df.shape[1]

    qG = np.array([0, 0, 0, 1])
    KM = np.zeros(data_row)

    start = time.time()

    camera = camera_info(df)

    gyroscope = quaternion_sensor([df["gyro_x"], df["gyro_y"], df["gyro_z"]])
    magnetometer = quaternion_sensor([df["mag_x"], df["mag_y"], df["mag_z"]])
    accelerator = quaternion_sensor([df["acc_x"], df["acc_y"], df["acc_z"]])

    print(time.time() - start)


    plt.plot([val.x for val in gyroscope.quaternion_signal])
    plt.show()


