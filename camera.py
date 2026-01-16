import configparser


class Camera:
    def __init__(self, df):
        self.config = configparser.ConfigParser()

        self.timestamp = list(df["Timestamp"])
        self.position_x = list(df["pos_x"])
        self.position_y = list(df["pos_y"])
        self.position_z = list(df["pos_z"])
        self.x = list(df["cam_qx"])
        self.y = list(df["cam_qy"])
        self.z = list(df["cam_qz"])
        self.w = list(df["cam_qw"])
        self.isTracked = list(df["isTracked"])

    def get_delta_t(self, i):
        return self.timestamp[i] - self.timestamp[i-1]
