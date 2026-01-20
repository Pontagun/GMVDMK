import numpy as np
import quaternion as qtn
import configparser
import helper


class Correction:
    def __init__(self, *args):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.x = args[0].x
        self.y = args[0].y
        self.z = args[0].z

        self.init_v = helper.get_vector(args[1])
        self.init_q = args[1]

    @staticmethod
    def get_qg_adjusted(qg, delta_qref):
        qg_ref = qg * delta_qref
        return qg_ref

    @staticmethod
    def get_mu_fusion(u, v):
        return (u + v) / 2

    @staticmethod
    def get_radian(u, v):
        u_mag = np.linalg.norm(u)
        v_mag = np.linalg.norm(v)

        dot = np.dot(u, v)
        res = dot / (u_mag * v_mag)
        res = np.clip(res, -1.0, 1.0)

        rad = np.arccos(res)

        return rad

    def get_delta_qref(self, v_sim):
        qw = self.get_qref_w(v_sim)
        qv = self.get_qref_v(v_sim)
        return qtn.as_quat_array([qw] + list(qv))

    def get_sim_reading_frame_body(self, q_rot):  # Change function name to something from seeing gravity vector from body frame.
        return q_rot.conjugate() * self.init_q * q_rot

    def get_qref_w(self, v_sim):
        reading_norm = np.linalg.norm([self.x, self.y, self.z])
        sim_norm = np.linalg.norm(qtn.as_float_array(v_sim))
        dot = np.dot([self.x, self.y, self.z], qtn.as_vector_part(v_sim))

        qref_w = reading_norm * sim_norm + dot

        return qref_w

    def get_qref_v(self, v_sim):
        return np.cross([self.x, self.y, self.z], qtn.as_vector_part(v_sim))

    def get_mu_ka(self, v):
        slope = int(self.config['SLOPE']["MuKa"])
        gamma = self.get_radian(self.init_v, v)
        r = (slope * gamma) + 1

        mu_ka = (1 + r + abs(1 + r)) / 4  # Best case, 1 - Worst cast, 0.

        return mu_ka

    def get_mu_km(self, v):
        v_magnitude = np.linalg.norm(v)
        compass_magnitude = np.linalg.norm(self.init_v)

        diff_mag = v_magnitude / compass_magnitude
        diff_ang = self.get_radian(self.init_v, v)

        penalty = diff_ang * diff_mag

        mu_km = 1 - penalty
        mu_km = (mu_km + abs(mu_km)) / 2

        return mu_km

    def get_mu_k(self, u, v):
        slope = int(self.config['SLOPE']["MuK"])

        mu_k = (v * slope) - slope + 1
        mu_k = (mu_k + abs(mu_k)) / 2
        return u * mu_k
