import configparser
import pandas as pd
import numpy as np

import helper
from sensor import QSensor
import matplotlib.pyplot as plt
from camera import Camera
import quaternion
from pipeline import Correction

import ahrs
from ahrs.filters import EKF, FKF, UKF, Tilt, AngularRate, Madgwick, Mahony, Complementary

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
    temp = [[], [], [], [], [], []]

    camera = Camera(df)
    gyro = QSensor(df["gyro_x"], df["gyro_y"], df["gyro_z"])
    magnet = QSensor(df["mag_x"], df["mag_y"], df["mag_z"])
    accel = QSensor(df["acc_x"], df["acc_y"], df["acc_z"])

    m_init = magnet.quat[0]
    a_init = accel.quat[0]

    alpha_mtnlns = 1.0
    mu_k_prelim = 0
    mu_k = 0

    g = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    a = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    m = df[["mag_x", "mag_y", "mag_z"]].to_numpy()

    g_ned = df[["gyro_z", "gyro_x", "gyro_y"]].to_numpy()
    a_ned = df[["acc_z", "acc_x", "acc_y"]].to_numpy()
    m_ned = df[["mag_z", "mag_x", "mag_y"]].to_numpy()

    # g_ned[:, 2] = -1 * g_ned[:, 2]
    g_ned = g
    a_ned[:, 2] = -1 * a_ned[:, 2]
    m_ned[:, 2] = -1 * m_ned[:, 2]

    ekf = EKF(gyr=g, acc=a, mag=m, frequency=120, magnetic_ref=m_ned[0], q0=[1, 0, 0, 0])

    # fkf = EKF(gyr=g, acc=a, mag=m, frequency=60)
    # ukf = UKF(gyr=g, acc=a, mag=m, frequency=120)
    tilt_ned = Tilt(mag=m_ned)
    angular_rate = AngularRate(gyr=g, frequency=120)
    angular_rate_ned = AngularRate(gyr=g_ned, frequency=120)
    madgwick = Madgwick(gyr=g, acc=a, mag=m, frequency=120, q0=[1, 0, 0, 0])
    mahony = Mahony(gyr=g, acc=a, mag=m, frequency=120)

    compli = Complementary(gyr=g_ned, frequency=100)
    # for i in range(alpha_window, data_row):
    #     # Get qG with no correction.
    #     delta_t = camera.get_delta_t(i) / 1000
    #
    #     qDot = .5 * (qG * gyro.quat[i])
    #     power = delta_t * qDot * qG.conjugate()
    #     qG = np.exp(power) * qG
    #
    #     # This qG has drift.
    #     qG = QSensor.get_quat_normalized(qG)
    #
    #     # Get alpha
    #     hist_accel = accel.quat[i - alpha_window]
    #     curr_accel = accel.quat[i]
    #     stillness_acc = helper.get_sensor_diff(hist_accel, curr_accel)
    #     alpha_mtnlns = helper.get_gamma_filter(stillness_acc, alpha_mtnlns)
    #
    #     a_pipeline = Correction(accel.quat[i], a_init)
    #     m_pipeline = Correction(magnet.quat[i], m_init)
    #
    #     a_qG = a_pipeline.get_sim_reading_frame_body(qG)
    #     qA_delta = a_pipeline.get_delta_qref(a_qG)
    #     qA_delta = QSensor.get_quat_normalized(qA_delta)
    #     qGA = a_pipeline.get_qg_adjusted(qG, qA_delta)
    #     qGA = QSensor.get_quat_normalized(qGA)
    #
    #     m_qG = m_pipeline.get_sim_reading_frame_body(qG)
    #     qM_delta = m_pipeline.get_delta_qref(m_qG)
    #     qM_delta = QSensor.get_quat_normalized(qM_delta)
    #     qGM = m_pipeline.get_qg_adjusted(qG, qM_delta)
    #     qGM = QSensor.get_quat_normalized(qGM)
    #
    #     # Single slerp.
    #     qSA = quaternion.slerp_evaluate(qG, qGA, alpha_mtnlns)
    #     qSM = quaternion.slerp_evaluate(qG, qGM, mu_k)
    #
    #     # Double slerp.
    #     qG = quaternion.slerp_evaluate(qSM, qSA, alpha_mtnlns)
    #
    #     qG = QSensor.get_quat_normalized(qG)
    #     qG_lst.append([qG.w, qG.x, qG.y, qG.z])
    #
    #     magnet_frame_inert_q = m_pipeline.get_sim_reading_frame_world(qG)
    #     magnet_frame_inert_v = helper.get_vector(magnet_frame_inert_q)
    #     mk_ka = m_pipeline.get_mu_ka(magnet_frame_inert_v)
    #     mk_km = m_pipeline.get_mu_km(magnet_frame_inert_v)
    #     mu_k_prelim = m_pipeline.get_mu_fusion(mk_ka, mk_km)
    #     mu_k = m_pipeline.get_mu_k(mu_k_prelim, alpha_mtnlns)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(g, linewidth=.5, label=['x', 'y', 'z'])
    ax1.legend(loc="upper right")
    ax2.plot(g_ned, linewidth=.5, label=['x', 'y', 'z'])
    ax2.legend(loc="upper right")
    # ax3.plot(m, linewidth=.5, label=['x', 'y', 'z'])
    # ax3.legend(loc="upper right")
    ax3.plot(madgwick.Q, linewidth=.5, label=['w', 'x', 'y', 'z'])
    ax3.legend(loc="upper right")
    ax4.plot(ekf.Q, linewidth=.5, label=['w', 'x', 'y', 'z'])
    ax4.legend(loc="upper right")

    plt.show()
