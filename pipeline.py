import numpy as np
import quaternion


class Correction:
    def __init__(self, m_init):
        self.m_init = m_init

    def simulate_mag(self, q):
        # q = quaternion.as_quat_array(q)
        return q.conjugate() * self.m_init * q

    def get_delta_qref(self, v_reading, v_sim):
        qw = self.get_qref_w(v_reading, v_sim)
        qv = self.get_qref_v(v_reading, v_sim)
        return qw + qv

    @staticmethod
    def get_qg_adjusted(qg, delta_qref):
        qg_ref = qg * delta_qref
        return qg_ref

    @staticmethod
    def get_qref_w(v_reading, v_sim):
        reading_norm = np.linalg.norm(v_reading)
        sim_norm = np.linalg.norm(v_sim)
        dot = np.dot(v_reading, v_sim)

        qref_w = reading_norm * sim_norm + dot

        return qref_w

    @staticmethod
    def get_qref_v(v_reading, v_sim):
        return np.cross(v_reading, v_sim)
