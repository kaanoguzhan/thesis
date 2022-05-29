# %% ─────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────────

import glob
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────
# Class Definitions
# ────────────────────────────────────────────────────────────


class AWAKE_DataLoader:
    def __init__(self, experiment_name, img_dims):
        self.experiment_name = experiment_name
        self.data = self.load_experiment_data(experiment_name, img_dims)

    def load_experiment_data(self, experiment_name, img_dims):
        data = []
        files = glob.glob(f'data/{experiment_name}/*.h5')
        for i in range(0, len(files)):
            f = files[i]
            ds = AWAKE_Data(f, img_dims)
            data.append(ds)
        return data


class AWAKE_Data:
    def __init__(self, data_path, img_dims):
        self.data_path = data_path
        self.data = self.load_data()
        self.streak_image = self.get_streak_image(img_dims)
        self.streak_stamp = self.get_streak_stamp()
        self.timestamp = self.get_timestamp()

    def load_data(self):
        data = h5py.File(self.data_path, 'r')
        return data

    def get_timestamp(self):
        a = np.array(self.data['AwakeEventInfo']['timestamp'])
        timestamp = a/1e9
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object

    def get_streak_image(self, img_dims):
        streak_image = (np.reshape(self.data['AwakeEventData/XMPP-STREAK/StreakImage/streakImageData'], img_dims))
        return streak_image

    def get_streak_stamp(self):
        streak_stamp = (self.data['AwakeEventData']['TT43.BPM.430010']['Acquisition'].attrs['acqStamp'])/1e9
        return streak_stamp

    def get_all_variables(self):
        # TODO not really the intended use of this function
        return list(self.data['AwakeEventData'])
