import numpy as np
import quaternion


class Correction:
    def __init__(self, args):
        self.x = args.x
        self.y = args.y
        self.z = args.z

    def get_delta_qref(self, v_reading, v_sim):
        qw = self.get_qref_w(v_reading, v_sim)
        qv = self.get_qref_v(v_reading, v_sim)
        return quaternion.as_quat_array([qw] + list(qv))

    @staticmethod
    def get_sim_reading_frame_body(q_init, q_rot):  # Change function name to something from seeing gravity vector from body frame.
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
