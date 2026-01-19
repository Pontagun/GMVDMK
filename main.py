import configparser
import pandas as pd
import statistics
import numpy as np
from pandas.core.array_algos.masked_reductions import mean

import helper
from sensor import QSensor
import matplotlib.pyplot as plt
from camera import Camera
import time
import quaternion
from pipeline import Correction

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    df = pd.read_csv(config['ENVIRONMENT']["Filename"])
    alpha_window = int(config['GMVDMK_WINDOW']['AlphaWindow'])

    df.columns = df.columns.str.replace(' ', '')
    data_row = df.shape[0]
    data_col = df.shape[1]

    qG = np.quaternion(1, 0, 0, 0)
    qG_lst = []
    temp = [[], []]

    camera = Camera(df)
    gyro = QSensor(df["gyro_x"], df["gyro_y"], df["gyro_z"])
    magnet = QSensor(df["mag_x"], df["mag_y"], df["mag_z"])
    accel = QSensor(df["acc_x"], df["acc_y"], df["acc_z"])

    m_init = magnet.quat[0]
    a_init = accel.quat[0]
    alpha_mtnlns = 0.0

    for i in range(alpha_window, data_row):
        # Get qG with no correction.
        delta_t = camera.get_delta_t(i) / 1000

        qDot = .5 * (qG * gyro.quat[i])
        power = delta_t * qDot * qG.conjugate()
        qG = np.exp(power) * qG
        qG = QSensor.get_quat_normalized(qG)

        # Get alpha
        v = accel.quat[i - alpha_window]
        w = accel.quat[i]
        stillness_acc = helper.get_sensor_diff(v, w)
        alpha_mtnlns = helper.get_gamma_filter(stillness_acc ** 2, alpha_mtnlns)

        a_correction = Correction(accel.quat[i], a_init)
        m_correction = Correction(magnet.quat[i], m_init)

        a_qG = a_correction.get_sim_reading_frame_body(a_init, qG)
        qA_delta = a_correction.get_delta_qref(a_correction, a_qG)
        qGA = a_correction.get_qg_adjusted(qG, qA_delta)

        m_qG = m_correction.get_sim_reading_frame_body(m_init, qG)
        qM_delta = m_correction.get_delta_qref(m_correction, m_qG)
        qGM = m_correction.get_qg_adjusted(qG, qM_delta)

        # Get Kmu
        # get_sim_reading_frame_body with a conjugate() quaternion is moving to the inertial perspective.
        magnet_frame_inert = m_correction.get_sim_reading_frame_body(magnet.quat[i], qG.conjugate())
        magnet_frame_inert = helper.get_vector(magnet_frame_inert)

        mk_ka = m_correction.get_mu_ka(magnet_frame_inert)
        mk_km = m_correction.get_mu_km(magnet_frame_inert)
        mk_merged = np.mean([mk_ka, mk_km])


        mu_k = m_correction.get_mu_k(mk_merged, alpha_mtnlns)

        qSA = quaternion.slerp_evaluate(qG, qGA, alpha_mtnlns)
        qSM = quaternion.slerp_evaluate(qG, qGM, mu_k)


        # qG = quaternion.slerp_evaluate(qSA, qSM, alpha_mtnlns)
        qSA = QSensor.get_quat_normalized(qSA)
        qSM = QSensor.get_quat_normalized(qSM)
        qG = quaternion.slerp_evaluate(qSA, qSM, alpha_mtnlns)

        qG_lst.append(gyro.quat[i])
        temp[0].append(mk_ka)
        temp[1].append(mk_km)


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot([val.x for val in qG_lst])
    ax1.plot([val.y for val in qG_lst])
    ax1.plot([val.z for val in qG_lst])
    # plt.plot([val.w for val in qG_lst])


    ax2.plot(temp[0], 'r')
    ax3.plot(temp[1], 'b')
    plt.show()
