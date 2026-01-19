import numpy as np
import quaternion
import configparser
import helper


class Correction:
    def __init__(self, *args):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.x = args[0].x
        self.y = args[0].y
        self.z = args[0].z

        self.init_vector = helper.get_vector(args[1])

    def get_delta_qref(self, v_reading, v_sim):
        qw = self.get_qref_w(v_reading, v_sim)
        qv = self.get_qref_v(v_reading, v_sim)
        return quaternion.as_quat_array([qw] + list(qv))

    @staticmethod
    def get_sim_reading_frame_body(q_init,
                                   q_rot):  # Change function name to something from seeing gravity vector from body frame.
        return q_rot.conjugate() * q_init * q_rot

    @staticmethod
    def get_qg_adjusted(qg, delta_qref):
        qg_ref = qg * delta_qref
        return qg_ref

    @staticmethod
    def get_qref_w(v_reading, v_sim):
        reading_norm = np.linalg.norm([v_reading.x, v_reading.y, v_reading.z])
        sim_norm = np.linalg.norm(quaternion.as_float_array(v_sim))
        dot = np.dot([v_reading.x, v_reading.y, v_reading.z], [v_sim.x, v_sim.y, v_sim.z])

        qref_w = reading_norm * sim_norm + dot

        return qref_w

    @staticmethod
    def get_qref_v(v_reading, v_sim):
        return np.cross([v_reading.x, v_reading.y, v_reading.z], [v_sim.x, v_sim.y, v_sim.z])

    def get_mu_ka(self, v):
        slope = int(self.config['SLOPE']["MuKa"])

        gamma = self.get_radian(self.init_vector, v)

        r = slope * gamma + 1

        mk_ka = (1 + r + abs(1 + r)) / 4  # Best case, 1 - Worst cast, 0.

        return mk_ka

    def get_mu_km(self, v):

        v_magnitude = np.linalg.norm(v)
        compass_magnitude = np.linalg.norm(self.init_vector)

        diff_mag = v_magnitude / compass_magnitude
        diff_ang = self.get_radian(self.init_vector, v)

        penalty = diff_ang * diff_mag

        mu_km = 1 - penalty
        mu_km = (mu_km + abs(mu_km)) / 2

        return mu_km

    @staticmethod
    def get_mu_fusion(u, v):
        return (u + v) / 2

    @staticmethod
    def get_radian(u, v):
        try:
            u = list(u)
            v = list(v)
        except TypeError:
            return None

        u_mag = np.linalg.norm(u)
        v_mag = np.linalg.norm(v)

        dot = np.dot(u, v)

        rad = np.arccos(dot / (u_mag * v_mag))

        return np.clip(rad, -1.0, 1.0)

    def get_mu_k(self, u, v):
        slope = int(self.config['SLOPE']["MuK"])

        mu_k1 = (v * slope) - slope + 1
        mu_k = (mu_k1 + abs(mu_k1)) / 2
        return u * mu_k
