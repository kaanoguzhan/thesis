# %% ─────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────────

import glob
from datetime import datetime

import h5py
import numpy as np

# ────────────────────────────────────────────────────────────
# Class Definitions
# ────────────────────────────────────────────────────────────


class AWAKE_DataLoader:
    def __init__(self, experiment_name, img_dims):
        self.__data = self.__load_experiment_data(experiment_name, img_dims)

    def __load_experiment_data(self, experiment_name, img_dims):
        data = []
        files = glob.glob(f'data/{experiment_name}/*.h5')
        for i in range(0, len(files)):
            f = files[i]
            ds = AWAKE_Image_Data(f, img_dims)
            data.append(ds)
        return data

    def get_data_at_index(self, index):
        return self.__data[index]

    def get_size(self):
        return len(self.__data)


class AWAKE_Image_Data:
    def __init__(self, data_path, img_dims=None):
        self.__img_dims = img_dims
        self.__data_path = data_path
        self.__data = self.__load_data()

    def __load_data(self):
        data = h5py.File(self.__data_path, 'r')
        return data

    def get_image(self):
        if self.__img_dims is None:
            raise ValueError('img_dims must be specified')
        streak_image = (np.reshape(self.__data['AwakeEventData']['XMPP-STREAK']['StreakImage']['streakImageData'], self.__img_dims))
        return streak_image

    def get_time_values(self):
        time_axis_length = self.__img_dims[0]
        time_values = self.__data['AwakeEventData']['XMPP-STREAK']['StreakImage']['streakImageTimeValues'][:time_axis_length]
        # fix the non-reported last time value
        time_values[-1] = time_values[-2] + (time_values[-2] - time_values[-3])
        return time_values

    def get_timestamp(self):
        a = np.array(self.__data['AwakeEventInfo']['timestamp'])
        timestamp = a/1e9
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object

    def get_image_dims(self):
        if self.__img_dims is None:
            raise ValueError('img_dims have not been specified')
        else:
            return self.__img_dims

    def get_streak_stamp(self):
        streak_stamp = (self.__data['AwakeEventData']['TT43.BPM.430010']['Acquisition'].attrs['acqStamp'])/1e9
        return streak_stamp

    def get_all_variables(self):
        # TODO not really the intended use of this class
        return list(self.__data['AwakeEventData'])
