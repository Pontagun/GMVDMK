import pandas as pd
import statistics
import configparser
import numpy as np
import quaternion

class quaternion_sensor:
    def __init__(self, coordinate):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        x = self.get_smoothen_signal(list(coordinate[0])) # column data
        y = self.get_smoothen_signal(list(coordinate[1]))
        z = self.get_smoothen_signal(list(coordinate[2]))
        w = [0 for i in range(len(coordinate[0]))]

        self.quaternion_signal = self.get_quaternion_signal(w, x, y, z)


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

    @staticmethod
    def get_quaternion_signal(w, x, y, z):
        quaternion_signal = []
        for index in range(len(w)):
            q = np.quaternion(w[index], x[index], y[index], z[index])
            quaternion_signal.append(q)

        return  quaternion_signal




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


