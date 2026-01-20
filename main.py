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

    m_init = QSensor.get_quat_normalized(magnet.quat[30])
    a_init = QSensor.get_quat_normalized(accel.quat[30])

    alpha_mtnlns = 1.0
    mu_k_prelim = 0.0
    mu_k = 0.0

    for i in range(alpha_window, data_row):
        a_pipeline = Correction(accel.quat[i], a_init)
        m_pipeline = Correction(magnet.quat[i], m_init)

        magnet_frame_inert_q = m_pipeline.get_sim_reading_frame_body(qG.conjugate())
        magnet_frame_inert_q = QSensor.get_quat_normalized(magnet_frame_inert_q)
        magnet_frame_inert_v = helper.get_vector(magnet_frame_inert_q)

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

        a_qG = a_pipeline.get_sim_reading_frame_body(qG)
        qA_delta = a_pipeline.get_delta_qref(a_qG)
        qA_delta = QSensor.get_quat_normalized(qA_delta)
        qGA = a_pipeline.get_qg_adjusted(qG, qA_delta)

        # m_qG = m_pipeline.get_sim_reading_frame_body(qG)
        # qM_delta = m_pipeline.get_delta_qref(m_qG)
        # qGM = m_pipeline.get_qg_adjusted(qG, qM_delta)

        # Single slerp.
        qSA = quaternion.slerp_evaluate(qGA, qGA, alpha_mtnlns)
        # qSM = quaternion.slerp_evaluate(qGA, qGA, mu_k)

        # Double slerp.
        qG = quaternion.slerp_evaluate(qSA, qSA, alpha_mtnlns)

        qG = QSensor.get_quat_normalized(qSA)
        qG_lst.append(magnet_frame_inert_v)

        # TODO: Exam and try to drop mk_km.
        # Get Kmu
        # get_sim_reading_frame_body with a conjugate() quaternion is moving to the inertial perspective.

        mk_ka = m_pipeline.get_mu_ka(magnet_frame_inert_v)
        mk_km = m_pipeline.get_mu_km(magnet_frame_inert_v)
        mu_k_prelim = np.mean([mk_ka, mk_km])
        mu_k = m_pipeline.get_mu_k(mu_k_prelim, alpha_mtnlns)

        temp[0].append(mk_ka)
        temp[1].append(mk_km)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot([val[0] for val in qG_lst], linewidth=.5)
    ax1.plot([val[1] for val in qG_lst], linewidth=.5)
    ax1.plot([val[2] for val in qG_lst], linewidth=.5)
    # ax1.plot([val.w for val in qG_lst], linewidth=.5)

    ax2.plot(temp[0], 'r', linewidth=.5)
    ax2.plot(temp[1], 'b', linewidth=.5)

    plt.show()
