# %% ─────────────────────────────────────────────────────────────────────────────
#  Imports, Constants, Settings
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
#  Imports
# ────────────────────────────────────────────────────────────
import scipy
import argparse
import copy
import json
import os
import sys
import warnings
from datetime import datetime
from scipy import interpolate

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.awake_data_loader import AWAKE_DataLoader
from utils.beam_utils import (find_marker_laser_pulse,
                              get_window_around_beam_centre)
from utils.general_utils import in_ipynb, merge_pdfs, natural_sort
from utils.image_utils import split_image_with_stride

# ─────────────────────────────────────────────────────────────
#  Default Constants
# ─────────────────────────────────────────────────────────────
BEAM_WINDOW = 100
SPLIT_WINDOW_SIZE = 10
SPLIT_STRIDE = 10
PSD_CUTOFF = (3, 0.05)
PADDING_MULTIPLIER = 50

# ─────────────────────────────────────────────────────────────
#  Settings
# ─────────────────────────────────────────────────────────────

print(f'Using Jupyter Notebook:{in_ipynb()}')
if in_ipynb():
    sys.argv = ['']
else:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.figsize': [15, 10],
    'savefig.transparent': False,
    'savefig.facecolor': 'w',
    'image.interpolation': 'none',
})

# Initilize arg parser
parser = argparse.ArgumentParser(description='FFT denoise analysis')
parser.add_help = True
parser.add_argument('--beam_window', type=int, default=BEAM_WINDOW, help='Beam Windows width around the beam center')
parser.add_argument('--split_window_size', type=int, default=SPLIT_WINDOW_SIZE, help='TODO')
parser.add_argument('--split_stride', type=int, default=SPLIT_STRIDE, help='TODO')
parser.add_argument('--psd_cutoff', type=float, default=PSD_CUTOFF, nargs='+', help='TODO')
parser.add_argument('--padding_multiplier', type=int, default=PADDING_MULTIPLIER, help='TODO')
args = parser.parse_args()

args.psd_cutoff = tuple(args.psd_cutoff)

# Set up log directory with timestamp and create file writer
current_time = datetime.now().strftime("%Y.%m.%d")
logdir = f'logs/Time_Axis_linearization/{current_time}'
file_writer_1 = SummaryWriter(f'{logdir}/tensorboard_logs')


print(f'\
Parameters:\n\
----------------------------------------\n\
    Log directory: {logdir}\n\
----------------------------------------\n\
')

#  ─────────────────────────────────────────────────────────────────────────────
#  Linear Time Axis vs Experiment Timestamps comparison
# ────────────────────────────────────────────────────────────────────────────────

# Load Image using AWAKE_DataLoader
adl = AWAKE_DataLoader('awake_1', [512, 672])
target_image_data = adl.get_data_at_index(3)

streak_img = target_image_data.get_image()
img_timevalues = target_image_data.get_time_values()

timeaxis = img_timevalues
timeax_lin = np.linspace(timeaxis[0], timeaxis[-1], len(timeaxis))

# Plot Time values
fig, axs = plt.subplots(figsize=(18, 12), nrows=2, ncols=1)
# add space between plots
plt.sca(axs[0])
for i in range(adl.get_size()):
    target_image_data = adl.get_data_at_index(i)
    experiment_name = target_image_data.get_experiment_name()
    img_timevalues = target_image_data.get_time_values()
    plt.plot(img_timevalues, label=f'{experiment_name}')
plt.plot(timeax_lin, color='black', label='Linear time')
plt.xticks(np.arange(0, len(img_timevalues), 30))
plt.title('Experiment Timestamps vs Linear Time')
plt.xlabel('Pixel index')
plt.ylabel('Timestamp (ps)')
plt.grid()
plt.legend()

plt.sca(axs[1])
for i in range(adl.get_size()):
    target_image_data = adl.get_data_at_index(i)
    experiment_name = target_image_data.get_experiment_name()
    img_timevalues = target_image_data.get_time_values()
    plt.plot(img_timevalues - timeax_lin, label=f'{experiment_name}')
plt.xticks(np.arange(0, len(img_timevalues), 30))
plt.title('Difference: Linear Time - Experiment Timestamps')
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig(f'{logdir}/1-Linear_Time_Axis_vs_Experiment_Timestamps.pdf')


# %%

x_value = np.array([0, 1, 2, 3, 4, 5])
y_value = np.array([0, 1.1, 2.2, 3.5, 3.9, 5])
y_lin = np.linspace(y_value[0], y_value[-1], len(y_value))
# y_value = np.array([50, 3, 4, 20, 160])

f_linear = interpolate.interp1d(y_value, x_value)
f_quad = interpolate.interp1d(y_value, x_value, kind="quadratic")

x_new = np.linspace(0, 5, 6)
x_new = np.array([0, 1.1, 2.2, 3.5, 3.9, 5])

plt.scatter(x_value, y_value, color='blue', label='distorted')
plt.scatter(x_value, y_lin, color='red', label='original')

plt.plot(x_value, f_linear(x_new), color='black', markersize=7, marker='.', label='linear')
# plt.plot(x_new, f_quad(x_new), 'o', color='green', markersize=3, label='quadratic')
plt.xlabel("X-Values")
plt.ylabel("Y-Values")
plt.title("1d Interpolation using scipy interp1d method")
plt.legend()
plt.show()


# %%
def streakimdef_old(im_init, timeaxis):  # e.g. streakimdef(h5filename,'XMPP-STREAK' (or 'TT41.BTV.412350.STREAK'),[])
    y = np.linspace(1, 672, 672)
    timeax_lin = np.linspace(timeaxis[0], timeaxis[510]+(timeaxis[510]-timeaxis[509]), len(timeaxis))
    # im_init_2d = np.rot90(np.reshape(im_init, (512, 672)))
    im_init_2d = im_init.T
    print(timeaxis.shape, y.size, im_init_2d.shape)
    im_interpol = interpolate.interp2d(timeaxis, y, im_init_2d)
    imstreak = im_interpol(timeax_lin, y)
    return imstreak.T


def streakimdef(im_init, timeaxis):  # e.g. streakimdef(h5filename,'XMPP-STREAK' (or 'TT41.BTV.412350.STREAK'),[])
    y = np.linspace(1, 512, 512)
    timeax_lin = np.linspace(timeaxis[0], timeaxis[510]+(timeaxis[510]-timeaxis[509]), len(timeaxis))
    # im_init_2d = np.rot90(np.reshape(im_init, (512, 672)))
    im_init = im_init.T
    im_init[im_init > 5000] = 5000
    
    # Normalize im_init
    im_init_max = np.max(im_init)
    im_init_min = np.min(im_init)
    im_init_norm = (im_init - im_init_min) / (im_init_max - im_init_min)

    print(len(timeax_lin), len(timeax_lin))
    print(im_init_norm.shape)

    # Normalize timeaxis
    timeaxis_norm = (timeaxis - timeaxis[0]) / (timeaxis[-1] - timeaxis[0])
    # Normalize timeax_lin
    timeax_lin_norm = (timeax_lin - timeax_lin[0]) / (timeax_lin[-1] - timeax_lin[0])

    f_linear = interpolate.interp1d(timeaxis_norm, timeax_lin_norm)

    # for each row in im_init_2d, interpolate the timeaxis to the timeaxis_lin
    im_interpol = np.zeros_like(im_init_norm)
    for i in range(im_init_norm.shape[0]):
        
        if i % 100 == 0:
            print(f'Interpolating row {i}')
            
        org_max = np.max(im_init[i])
        org_min = np.min(im_init[i])

        # print(f'{start} - {end}')

        # Scale all values between start and end range
        im_interpol[i] = (im_init[i] - org_min) / (org_max - org_min)

        im_interpol[i] = f_linear(im_init_norm[i])

        # Reverse scaling to original range
        im_interpol[i] = (im_interpol[i] * (org_max - org_min)) + org_min

    # Reverse normalization of im_init_norm
    im_interpol = (im_interpol * (im_init_max - im_init_min)) + im_init_min


    imstreak = im_interpol.T
    return imstreak


test_img = (np.ones((512, 672)).T * np.linspace(0, 5000, 512)).T
# test_img = (np.ones((512, 672)) * np.linspace(0,5000,672))
# test_img = adl.get_data_at_index(3).get_image()


intp_img = copy.deepcopy(test_img)
intp_img = streakimdef_old(intp_img, timeaxis)

# Plot original image
fig = plt.figure(figsize=(13, 13))
plt.imshow(test_img, cmap='gray', vmin=0, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)

# Plot the interpolated image
fig = plt.figure(figsize=(13, 13))
# plt.imshow(intp_img-test_img, cmap='coolwarm', vmin=-5e-12, vmax=5e-12)
plt.imshow(intp_img-test_img, cmap='coolwarm')
# plt.imshow(intp_img-test_img, cmap='coolwarm', vmin=-3000, vmax=3000)
# plt.imshow(intp_img-test_img, cmap='coolwarm', vmin=-5000, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.show()

# Plot the difference between the original and the interpolated image
fig = plt.figure(figsize=(13, 13))
# plt.imshow(intp_img-test_img, cmap='coolwarm', vmin=-3000, vmax=3000)
plt.imshow(intp_img, cmap='gray', vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
# plt.imshow(intp_img-test_img, cmap='viridis', alpha=0.2)
plt.show()

# %%
