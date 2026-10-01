import configparser
import pandas as pd
import numpy as np

import helper
from sensor import QSensor
import matplotlib.pyplot as plt
from camera import Camera
import quaternion
from pipeline import Correction

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    df = pd.read_csv(config['ENVIRONMENT']["Filename"])
    alpha_window = int(config['GMVDMK_WINDOW']['AlphaWindow'])
    quat_smooth_window = int(config['GMVDMK_WINDOW']['QuatSmoothWindow'])

    df.columns = df.columns.str.replace(' ', '')
    data_row = df.shape[0]
    data_col = df.shape[1]

    qG = np.quaternion(1, 0, 0, 0)
    qG_lst = []
    mus = []
    temp = [[], [], [], [], [], []]

    camera = Camera(df)
    gyro = QSensor(df["gyro_x"], df["gyro_y"], df["gyro_z"])
    magnet = QSensor(df["mag_x"], df["mag_y"], df["mag_z"])
    accel = QSensor(df["acc_x"], df["acc_y"], df["acc_z"])

    m_init = magnet.quat[0]
    a_init = accel.quat[0]

    a_pipeline = Correction(a_init)
    m_pipeline = Correction(m_init)

    # Stillness depends only on raw accel data, so compute it for every step up front.
    stillness_acc = helper.get_sensor_diff(accel.quat, alpha_window)
    alpha_mtnlns_lst = helper.get_gamma_filter(stillness_acc)

    mu_k_prelim = 0
    mu_k = 0

    for i in range(alpha_window, data_row):
        # Get qG with no correction.
        delta_t = camera.get_delta_t(i) / 1000

        qG = qG * np.exp(.5 * delta_t * gyro.quat[i])

        # This qG has drift.
        qG = QSensor.get_quat_normalized(qG)

        alpha_mtnlns = alpha_mtnlns_lst[i - alpha_window]

        # qG and the deltas are unit quaternions, so qGA / qGM need no further normalization.
        a_qG = a_pipeline.get_sim_reading_frame_body(qG)
        qA_delta = a_pipeline.get_delta_qref(accel.quat[i], a_qG)
        qA_delta = QSensor.get_quat_normalized(qA_delta)
        qGA = a_pipeline.get_qg_adjusted(qG, qA_delta)

        m_qG = m_pipeline.get_sim_reading_frame_body(qG)
        qM_delta = m_pipeline.get_delta_qref(magnet.quat[i], m_qG)
        qM_delta = QSensor.get_quat_normalized(qM_delta)
        qGM = m_pipeline.get_qg_adjusted(qG, qM_delta)

        # Single slerp.
        qSA = quaternion.slerp_evaluate(qG, qGA, alpha_mtnlns)
        qSM = quaternion.slerp_evaluate(qG, qGM, mu_k)

        # Double slerp.
        qG = quaternion.slerp_evaluate(qSM, qSA, alpha_mtnlns)

        qG = QSensor.get_quat_normalized(qG)
        qG_lst.append(qG)

        magnet_frame_inert_q = m_pipeline.get_sim_reading_frame_world(magnet.quat[i], qG)
        magnet_frame_inert_v = helper.get_vector(magnet_frame_inert_q)
        magnet_gamma = m_pipeline.get_radian(magnet_frame_inert_v)
        mk_ka = m_pipeline.get_mu_ka(magnet_gamma)
        mk_km = m_pipeline.get_mu_km(magnet_frame_inert_v, magnet_gamma)
        mu_k_prelim = m_pipeline.get_mu_fusion(mk_ka, mk_km)
        mu_k = m_pipeline.get_mu_k(mu_k_prelim, alpha_mtnlns)

        mus.append(mu_k)

    qG_lst = helper.get_quat_moving_average(qG_lst, quat_smooth_window)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot([round(val.x, 4) for val in qG_lst], linewidth=1)
    ax1.plot([round(val.y, 4) for val in qG_lst], linewidth=1)
    ax1.plot([round(val.z, 4) for val in qG_lst], linewidth=1)
    ax1.plot([round(val.w, 4) for val in qG_lst], linewidth=1)

    ax2.plot(alpha_mtnlns_lst, linewidth=1)
    ax3.plot(mus, linewidth=1)

    plt.show()
