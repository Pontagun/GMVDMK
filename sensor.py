import pandas as pd
import statistics
import configparser
import numpy as np
import quaternion


class QSensor:
    def __init__(self, *args):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        w = [0 for i in range(len(args[0]))]
        x = self.get_smoothen_signal(np.array(args[0]))  # column data
        y = self.get_smoothen_signal(np.array(args[1]))
        z = self.get_smoothen_signal(np.array(args[2]))
        self.quat = self.get_quat(w, x, y, z)

    def get_smoothen_signal(self, axis):
        window = int(self.config['GMVDMK_WINDOW']["SmoothWindow"])
        signal = []

        for index, value in enumerate(axis):
            if index > window:
                signal.append(round(statistics.mean(axis[index - window:index]), 4))
            else:
                signal.append(round(value, 4))

        return signal

    @staticmethod
    def get_quat_normalized(q):
        if type(q) != quaternion.quaternion:
            return q

        norm_q = np.linalg.norm(quaternion.as_float_array(q))
        unit_q = q / norm_q
        unit_q.w = round(unit_q.w, 4)
        unit_q.x = round(unit_q.x, 4)
        unit_q.y = round(unit_q.y, 4)
        unit_q.z = round(unit_q.z, 4)
        return unit_q

    @staticmethod
    def get_quat(w, x, y, z):
        q = []
        for index in range(len(w)):
            q.append(quaternion.as_quat_array([w[index], x[index], y[index], z[index]]))

        return np.asarray(q)
