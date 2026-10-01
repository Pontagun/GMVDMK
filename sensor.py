import configparser
import numpy as np
import quaternion


class QSensor:
    def __init__(self, *args):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        w = np.zeros(len(args[0]))
        x = self.get_smoothen_signal(np.array(args[0]))  # column data
        y = self.get_smoothen_signal(np.array(args[1]))
        z = self.get_smoothen_signal(np.array(args[2]))
        self.quat = self.get_quat(w, x, y, z)

    def get_smoothen_signal(self, axis):
        window = int(self.config['GMVDMK_WINDOW']["SmoothWindow"])
        signal = axis.astype(float)

        # Mean of the previous `window` samples (current sample excluded); the first `window` samples are left as-is.
        if len(axis) > window:
            signal[window:] = np.lib.stride_tricks.sliding_window_view(axis, window)[:-1].mean(axis=1)

        return signal

    @staticmethod
    def get_quat_normalized(q):
        if type(q) != quaternion.quaternion:
            return q

        return q.normalized()

    @staticmethod
    def get_quat(w, x, y, z):
        return quaternion.as_quat_array(np.column_stack([w, x, y, z]))
