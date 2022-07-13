# %% ─────────────────────────────────────────────────────────────────────────────
#  Imports, Constants, Settings
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
#  Imports
# ────────────────────────────────────────────────────────────
import argparse
import copy
import json
import os
import sys
import warnings
from datetime import datetime
import itertools
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
BEAM_WINDOW = 300
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
parser.add_argument('--beam_window', type=int, default=BEAM_WINDOW, help='Beam Windows width around the beam centre')
parser.add_argument('--split_window_size', type=int, default=SPLIT_WINDOW_SIZE, help='TODO')
parser.add_argument('--split_stride', type=int, default=SPLIT_STRIDE, help='TODO')
parser.add_argument('--psd_cutoff', type=float, default=PSD_CUTOFF, nargs='+', help='TODO')
parser.add_argument('--padding_multiplier', type=int, default=PADDING_MULTIPLIER, help='TODO')
args = parser.parse_args()

args.psd_cutoff = tuple(args.psd_cutoff)

# Set up log directory with timestamp and create file writer
current_time = datetime.now().strftime("%Y.%m.%d")
logdir = f'logs/AWAKE_PSD_Peak_Analysis/{current_time}' +\
    f'_beamwin{args.beam_window:>03d}' +\
    f'_splitwin{args.split_window_size:>02d}' +\
    f'_stride{args.split_stride:>02d}' +\
    f'_psdcut{args.psd_cutoff[0]:>.2f}-{args.psd_cutoff[1]:>.4f}' +\
    f'_pad{args.padding_multiplier:>02d}'
file_writer_1 = SummaryWriter(f'{logdir}/tensorboard_logs')


print(f'\
Parameters:\n\
----------------------------------------\n\
    Beam window: {args.beam_window}\n\
    Split window size: {args.split_window_size}\n\
    Split stride: {args.split_stride}\n\
    PSD Cutoff: {args.psd_cutoff}\n\
    0-Padding: {args.padding_multiplier}\n\
    Log directory: {logdir}\n\
----------------------------------------\n\
')

# %% ─────────────────────────────────────────────────────────────────────────────
#  Image loading
# ────────────────────────────────────────────────────────────────────────────────

# Load Image using AWAKE_DataLoader
adl = AWAKE_DataLoader('awake_1', [512, 672])
streak_img = adl.data[0].streak_image

# plot streak_img as a sanity check
plt.figure()
plt.imshow(streak_img, interpolation=None, vmax=5000)
plt.savefig(f'{logdir}/0_streak_img.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
print(f'Streak image shape: {streak_img.shape}')

streak_img = streak_img.T

# Crop image from the beginning of Marker Laser Pulse
mlp_start, mlp_end = find_marker_laser_pulse(streak_img, x_axis_is_space=True)
print(f'Marker laser pulse is between: {mlp_start, mlp_end}')
# streak_img = streak_img[:, 300:]
streak_img = streak_img[:, mlp_start:]
# plot the streak_img after the Marker Laser Pulse cutting as a sanity check
plt.figure()
plt.imshow(streak_img, vmax=5000)
plt.savefig(f'{logdir}/1_streak_img_after_mlp_cut.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# Cut a window around the beam centre and polot it as a sanity check
beam_window_start, beam_window_end = get_window_around_beam_centre(streak_img, args.beam_window, x_axis_is_space=True)
# beam_centre = streak_img[290:300, :]
beam_centre = streak_img[beam_window_start:beam_window_end, :]
plt.figure(figsize=(15, 10))
plt.imshow(beam_centre, vmax=5000)
plt.savefig(f'{logdir}/2_streak_img_after_mlp_cut_beam_window.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

tmp_beam_centre = copy.deepcopy(beam_centre)


# %% ------------------------------------------------------------------------------
# Handcrafted Covariance

def covariance_1D(x, y):
    x_mean, y_mean = x.mean(), y.mean()
    return np.sum((x - x_mean)*(y - y_mean))/np.sqrt(np.sum((x - x_mean)**2)*np.sum((y - y_mean)**2))


def covariance_2D(x, y):
    print(f'Length of x: {x.shape[0]} - Length of y: {y.shape[0]}')
    counter = 0
    covs = []
    for x_l, y_l in list(itertools.product(x, y)):
        cov = covariance_1D(x_l, y_l)
        covs.append(cov)

        counter += 1
        if counter % 10000 == 0:
            print(f'Processing... {counter}/{x.shape[0]*y.shape[0]}')

    print(f'Counter: {counter}')

    covs = np.array(covs)
    covs = covs.reshape(x.shape[0], y.shape[0])

    return covs
sy
# Pixel merging using Averaging
def merge_pixels_by_avg(image, window_size):
    ws = window_size
    new_x, new_y = image.shape[0]//ws, image.shape[1]//ws
    image_merge = np.zeros((new_x, new_y))

    for i in np.arange(0, new_x):
        for j in np.arange(0, new_y):
            # image_merge[i, j] = np.mean(image[i*ws:i*ws+ws, j*ws:j*ws+ws])
            # image_merge[i, j] = np.max(image[i*ws:i*ws+ws, j*ws:j*ws+ws])
            image_merge[i, j] = image[i*ws, j*ws]
    return image_merge

# %%
# Calculate and Plot Covariances

TYPE = 'first'

merge_by_8 = merge_pixels_by_avg(beam_centre, 8)

covariance_by_1 = covariance_2D(beam_centre, beam_centre)
covariance_by_8 = covariance_2D(merge_by_8, merge_by_8)

# Plot and save Downsampled Beam Images
plt.figure(figsize=(15, 10))
plt.imshow(beam_centre, vmax=5000)
plt.savefig(f'{logdir}/3_merge_by_1.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.figure(figsize=(15, 10))
plt.imshow(merge_by_8, vmax=5000)
plt.savefig(f'{logdir}/3_merge_by_8_{TYPE}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# Plot Covariance for image with 1 pixel merging
plt.figure(figsize=(15, 10))
plt.imshow(covariance_by_1, vmax=1, vmin=-1)
cbar = plt.colorbar()
cbar.set_label('Covariance Coefficient')
plt.xlabel('Row Number')
plt.ylabel('Row Number')
plt.savefig(f'{logdir}/4_covariance_by_1.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# Plot Covariance for image with 8 pixel merging
plt.figure(figsize=(15, 10))
plt.imshow(covariance_by_8, vmax=1, vmin=-1)
cbar = plt.colorbar()
cbar.set_label('Covariance Coefficient')
plt.xlabel('Row Number')
plt.ylabel('Row Number')
plt.savefig(f'{logdir}/4_covariance_by_8_{TYPE}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)


# %%
def covariances_by_distance(covs):
    covs_dist = []
    for dist in range(covs.shape[0]):
        dist_arr = []
        for idx in range(covs.shape[0]):
            covs_double = np.concatenate((covs[idx], covs[idx]), axis=0)
            cov_l = covs_double[idx - dist]
            cov_r = covs_double[idx + dist]
            dist_arr.append(cov_l)
            dist_arr.append(cov_r)
        covs_dist.append(dist_arr)
    return covs_dist


calc_dist_covs_1 = covariances_by_distance(covariance_by_1)
calc_dist_covs_8 = covariances_by_distance(covariance_by_8)


# Plot calc_dist_covs_1 using matplotlib Boxplots
plt.figure(figsize=(15, 10))
plt.boxplot(calc_dist_covs_1, showfliers=False)
plt.grid(axis='both', linestyle='--', color='0.75')
plt.title('Covariance Coefficients of Rows by Row Distance - No Image downsampling')
plt.xlabel('Row Distance')
plt.ylabel('Covariance Coefficient')
plt.xticks(np.arange(1, len(calc_dist_covs_1)+1, 5), np.arange(0, len(calc_dist_covs_1), 5))
# plt.xlim(0,100)
plt.ylim(-1.05, 1.05)
plt.savefig(f'{logdir}/5_covariances_by_distance_downsaple_1_all.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# Plot calc_dist_c8 using matplotlib Boxplots
plt.figure(figsize=(15, 10))
plt.boxplot(calc_dist_covs_8, showfliers=False)
plt.grid(axis='both', linestyle='--', color='0.75')
plt.title(f'Covariance Coefficients of Rows by Row Distance - Downsampling 8 ({TYPE})')
plt.xlabel('Row Distance')
plt.ylabel('Covariance Coefficient')
plt.xticks(np.arange(1, len(calc_dist_covs_8)+1, 1), np.arange(0, len(calc_dist_covs_8), 1))
plt.ylim(-1.05, 1.05)
plt.savefig(f'{logdir}/5_covariances_by_distance_downsaple_8_{TYPE}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

