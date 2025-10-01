import pandas as pd
import statistics
import configparser

class quaternion_sensor:
    def __init__(self, coordinate):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.x = self.smoothening_signal(coordinate[0]) # column data
        self.y = self.smoothening_signal(coordinate[1])
        self.z = self.smoothening_signal(coordinate[2])
        self.w = pd.Series([0 for i in range(len(self.x))])


    def smoothening_signal(self, axis):
        window = int(self.config['GMVDMK_WINDOW']["SmoothWindow"])
        signal = []

        for index, e in enumerate(axis):
            if index > window:
                signal.append(statistics.mean(axis.iloc[index - window:index]))

        return pd.Series(signal) # column data

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


