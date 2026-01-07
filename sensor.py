import pandas as pd
import statistics
import configparser
import numpy as np
import quaternion


class quaternion_sensor:
    def __init__(self, coordinate):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        w = [0 for i in range(len(coordinate[0]))]
        x = self.get_smoothen_signal(list(coordinate[0]))  # column data
        y = self.get_smoothen_signal(list(coordinate[1]))
        z = self.get_smoothen_signal(list(coordinate[2]))

        self.quaternion_signal = self.get_q_terms(w, x, y, z)

    def get_smoothen_signal(self, axis):
        window = int(self.config['GMVDMK_WINDOW']["SmoothWindow"])
        signal = []

        for index, value in enumerate(axis):
            if index > window:
                signal.append(statistics.mean(axis[index - window:index]))
            else:
                signal.append(value)

        return signal
        # return pd.Series(signal) # column data

    # def get_norm(self, i):
    #     return np.linalg.norm(self.quaternion_signal[i])
        # return np.sqrt(self.quaternion_signal[i].x ** 2 + self.quaternion_signal[i].y ** 2
        #                + self.quaternion_signal[i].z ** 2 + self.quaternion_signal[i].w ** 2)

    def get_unit_q(self, q):
        norm_q = np.linalg.norm(q)
        unit_q = q / norm_q

        return unit_q

    def get_q_terms(self, w, x, y, z):
        norm_q_terms = []
        for index in range(len(w)):
            unit_q = self.get_unit_q([w[index], x[index], y[index], z[index]])
            norm_q_terms.append(quaternion.as_quat_array(unit_q))

        return norm_q_terms

    # def get_diff_vectors(self, vector):
    #     # var:vector is a slice from the full list.
    #     vector_len = int(self.config['GMVDMK_WINDOW']["Window"])
    #
    #     if len(vector) < vector_len:
    #         return 1 # By assumption that the sensor is steady at the beginning.
    #
    #     delta = vector[-1] - vector[0]
    #
    #     max_axis_diff = max(delta)
    #
    #     return # float
