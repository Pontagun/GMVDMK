import configparser


class camera_info:
    def __init__(self, df):
        self.config = configparser.ConfigParser()

        self.timestamp = list(df["Timestamp"])
        self.x = list(df["pos_x"])
        self.y = list(df["pos_y"])
        self.z = list(df["pos_z"])
        self.camera_x = list(df["cam_qx"])
        self.camera_y = list(df["cam_qy"])
        self.camera_z = list(df["cam_qz"])
        self.camera_w = list(df["cam_qw"])
        self.isTracked = list(df["isTracked"])