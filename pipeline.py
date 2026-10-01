import math

import numpy as np
import quaternion
import configparser
import helper

config = configparser.ConfigParser()
config.read('config.ini')
slope_mu_ka = int(config['SLOPE']["MuKa"])
slope_mu_k = int(config['SLOPE']["MuK"])


class Correction:
    def __init__(self, init_q):
        # init_q is the sensor's first reading, used as the reference for every later reading.
        self.init_v = helper.get_vector(init_q)
        self.init_q = init_q
        self.init_norm = abs(init_q)

    @staticmethod
    def get_qg_adjusted(qg, delta_qref):
        qg_ref = qg * delta_qref
        return qg_ref

    @staticmethod
    def get_mu_fusion(u, v):
        return (u + v) / 2

    def get_radian(self, v):
        v_mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

        dot = self.init_v[0] * v[0] + self.init_v[1] * v[1] + self.init_v[2] * v[2]
        res = dot / (self.init_norm * v_mag)
        res = min(max(res, -1.0), 1.0)

        rad = math.acos(res)

        return rad

    @staticmethod
    def get_delta_qref(q_reading, q_sim):
        # Rotation from q_reading to q_sim: w = |a||b| + a.b, v = a x b.
        # For pure quaternions a * b = (-a.b, a x b), so one product gives both terms.
        p = q_reading * q_sim
        return np.quaternion(abs(q_reading) * abs(q_sim) - p.w, p.x, p.y, p.z)

    def get_sim_reading_frame_body(self, q_rot):
        # Change function name to something from seeing gravity vector from body frame.
        q = q_rot.conjugate() * self.init_q * q_rot
        # Drop the w part (rounding noise) so the result is a pure quaternion.
        return np.quaternion(0, q.x, q.y, q.z)

    @staticmethod
    def get_sim_reading_frame_world(q_reading, q_rot):
        return q_rot * q_reading * q_rot.conjugate()

    @staticmethod
    def get_mu_ka(gamma):
        r = 1 + slope_mu_ka * gamma
        # r = 1 (gamma = 0) means no difference between calculation and actual readings.
        mu_ka = (1 + r + abs(1 + r)) / 4  # Best case, 1 - Worst cast, negative number.

        return mu_ka

    def get_mu_km(self, v, diff_ang):
        v_magnitude = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

        diff_mag = v_magnitude / self.init_norm

        # diff_mag = 1 means same magnitude, so penalize its distance from 1.
        penalty = diff_ang + diff_mag

        mu_km = 1 - penalty
        mu_km = (mu_km + abs(mu_km)) / 2

        return mu_km

    @staticmethod
    def get_mu_k(temp_km, alpha):
        speed = (alpha * slope_mu_k) - slope_mu_k + 1
        speed = (speed + abs(speed)) / 2
        return temp_km * speed
