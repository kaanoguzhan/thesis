# %% ─────────────────────────────────────────────────────────────────────────────
# Imports, Constants, Settings
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────
import argparse
import io
import os
import sys
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from torch.utils.tensorboard import SummaryWriter

from utils.awake_data_loader import AWAKE_DataLoader
from utils.beam_utils import (find_marker_laser_pulse,
                              get_window_around_beam_center)
from utils.general_utils import in_ipynb
from utils.image_utils import split_image_with_stride

if in_ipynb():
    sys.argv = ['']
else:
    matplotlib.use('Agg')

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_BEAM_WINDOW = 100
SPLIT_WINDOW_SIZE = 10
SPLIT_STRIDE = 10
PSD_CUTOFF = (0.9, 0.01)
PADDING_MULTIPLIER = 5

print(f'Using Jupyter Notebook:{in_ipynb()}')

# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.figsize': [15, 10],
    'savefig.transparent': False,
    'savefig.facecolor': 'w'
})

# Initilize arg parser
parser = argparse.ArgumentParser(description='FFT denoise analysis')
parser.add_help = True
parser.add_argument('--beam_window', type=int, default=DEFAULT_BEAM_WINDOW, help='Beam Windows width around the beam center')
parser.add_argument('--split_window_size', type=int, default=SPLIT_WINDOW_SIZE, help='TODO')
parser.add_argument('--split_stride', type=int, default=SPLIT_STRIDE, help='TODO')
parser.add_argument('--psd_cutoff', type=int, default=PSD_CUTOFF, help='TODO')
parser.add_argument('--padding_multiplier', type=int, default=PADDING_MULTIPLIER, help='TODO')
args = parser.parse_args()

# Sets up log directory with timestamp
# current_time = datetime.now().strftime("%Y.%m.%d-%H_%M_%S")
current_time = datetime.now().strftime("%Y.%m.%d")
logdir = f'logs/{current_time}_AWAKE_PSD_Analysis' +\
    f'_window{args.beam_window}' +\
    f'_split{SPLIT_WINDOW_SIZE}-{SPLIT_STRIDE}' +\
    f'_PSDCutoff{args.psd_cutoff[0]}-{args.psd_cutoff[1]}' +\
    f'_Padding{args.padding_multiplier}'
# Creates a file writer for the log directory.
file_writer_1 = SummaryWriter(logdir)
# file_writer_2 = tf.summary.create_file_writer(logdir)


print(f'\
    Parameters:\n\
    ----------------------------------------\n\
    Beam window: {args.beam_window}\n\
    Split window size: {SPLIT_WINDOW_SIZE}\n\
    Split stride: {SPLIT_STRIDE}\n\
    PSD Cutoff: {args.psd_cutoff}\n\
    0-Padding: {args.padding_multiplier}\n\
    Log directory: {logdir}\n\
    ----------------------------------------\n\
    ')

# %% ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ────────────────────────────────────────────────────────────────────────────────

# Load Image using AWAKE_DataLoader
adl = AWAKE_DataLoader('awake_1', [512, 672])
streak_img = adl.data[0].streak_image

# plot streak_img as a sanity check
plt.figure()
plt.imshow(streak_img, vmax=5000)
print(f'Streak image shape: {streak_img.shape}')

streak_img = streak_img.T

# Crop image from the beginning of Marker Laser Pulse
mlp_start, mlp_end = find_marker_laser_pulse(streak_img, x_axis_is_space=True)
print(f'Marker laser pulse is between: {mlp_start, mlp_end}')
streak_img = streak_img[:, mlp_start:]
# plot the streak_img after the Marker Laser Pulse cutting as a sanity check
plt.figure(figsize=(15, 10))
plt.imshow(streak_img, vmax=5000)

# Cut a window around the beam center and polot it as a sanity check
beam_window_start, beam_window_end = get_window_around_beam_center(streak_img, args.beam_window, x_axis_is_space=True)
beam_center = streak_img[beam_window_start:beam_window_end, :]
plt.figure(figsize=(15, 10))
plt.imshow(beam_center, vmax=5000)

tmp_beam_center = beam_center.copy()
psd_cutoff = 0.1


# Split image into N parts
N = 10
# img_split = np.array_split(tmp_beam_center, N, axis=0)
img_splits = split_image_with_stride(image=tmp_beam_center, window_size=SPLIT_WINDOW_SIZE, stride=SPLIT_STRIDE)

# %%
# Display splitted images
for img_split in img_splits:
    plt.figure(figsize=(15, 10))
    plt.imshow(img_split, vmax=5000)

# %%


def fft_filter_img(img, psd_cutoff, fft_bin_multiplier, plot=False, return_plot=False):
    """
    Returns a fft filtered image.
    """

    # Save image shape and row by sum on the first dimension
    img_shape = img.shape
    signal = np.sum(img, axis=0)

    # Normalize signal between 0 and 1
    signal_max = np.max(signal)
    signal_min = np.min(signal)
    signal = (signal - signal_min) / (signal_max - signal_min)
    signal_org = signal.copy()

    t_exp = np.arange(len(signal))

    # Compute Fourier Coefficients of the f_sample signal
    # Fourier Coefficients have magnitude and phase
    Fn = np.fft.fft(signal, len(signal)*fft_bin_multiplier)

    # Use PSD to filter out noise

    # Compute Power Spectral Denisty
    p_s_d = Fn * np.conj(Fn) / len(signal)

    # Normalize PSD
    p_s_d_min = np.min(p_s_d)
    p_s_d_max = np.max(p_s_d)
    p_s_d = (p_s_d - p_s_d_min) / (p_s_d_max - p_s_d_min)

    # Find all frequencies with large power
    # Check if psd_cutoff is a number or a tuple
    if isinstance(psd_cutoff, tuple):
        psd_cutoff_max = psd_cutoff[0]
        psd_cutoff_min = psd_cutoff[1]
    else:
        psd_cutoff_max = np.inf
        psd_cutoff_min = psd_cutoff
    indices = np.logical_and(p_s_d > psd_cutoff_min, p_s_d < psd_cutoff_max)
    # indices = np.zeros((len(p_s_d),), dtype=bool)

    # Zero out smaller Fourier coefficients and their corresponding frequencies
    # Boradcast Fn to the same shape as img_shape
    p_s_d_clean = p_s_d * indices

    # Clean p_s_d and p_s_d_Clean from imaginary part
    p_s_d_clean = np.real(p_s_d_clean)
    p_s_d = np.real(p_s_d)

    # Go over each row on image and replace with PSD filtered signal
    for i in range(img_shape[0]):
        signal_row = img[i]
        Fn_row = np.fft.fft(signal_row, len(signal_row)*fft_bin_multiplier)
        Fn_row = Fn_row * indices
        img_ifft = np.fft.ifft(Fn_row)
        img[i] = np.real(img_ifft[:img_shape[1]])

    signal_clean = np.sum(img, axis=0)
    signal_clean_max = np.max(signal_clean)
    signal_clean_min = np.min(signal_clean)
    signal_clean = (signal_clean - signal_clean_min) / (signal_clean_max - signal_clean_min)

    # t_exp = np.arange(len(signal)/2, dtype=np.int32)

    if plot:
        # Plotting denoising results
        fig, axs = plt.subplots(4, 1, figsize=(20, 22))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.2)

        plt.sca(axs[0])
        plt.plot(t_exp, signal_org, color='k')
        plt.plot(t_exp, signal_org, 'o', color='k', label='Noisy signal')
        plt.xlim(t_exp[0], t_exp[-1])
        plt.title('Noisy signal')
        plt.ylim(0, 1)
        plt.grid()
        plt.legend()

        plt.sca(axs[1])
        plt.plot(t_exp, p_s_d[t_exp], color='b')
        plt.plot(t_exp, p_s_d[t_exp], 'o', color='b', linewidth=2, label='Noisy signal')
        plt.plot([t_exp[0], t_exp[-1]], [psd_cutoff_min, psd_cutoff_min], '--', color='tab:orange', label='PSD cutoff')
        plt.plot([t_exp[0], t_exp[-1]], [psd_cutoff_max, psd_cutoff_max], '--', color='tab:orange', label='PSD cutoff')
        plt.xlim(t_exp[0], t_exp[-1])
        # plt.xlim(freq[0], 40)
        # plt.ylim(0, psd_cutoff*10)
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        plt.title(f'Power Spectral Density')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectrum |Fn|^2')
        plt.legend()

        plt.sca(axs[2])
        plt.plot(t_exp, p_s_d[t_exp], color='b')
        plt.plot(t_exp, p_s_d_clean[t_exp], color='darkred')
        plt.plot(t_exp, p_s_d[t_exp], 'o', color='b', linewidth=2, label='Noisy signal')
        plt.plot(t_exp, p_s_d_clean[t_exp], 'o', color='darkred', linewidth=2, label='Filtered signal')
        plt.plot([t_exp[0], t_exp[-1]], [psd_cutoff_min, psd_cutoff_min], '--', color='tab:orange', label='PSD cutoff')
        plt.plot([t_exp[0], t_exp[-1]], [psd_cutoff_max, psd_cutoff_max], '--', color='tab:orange', label='PSD cutoff')
        plt.xlim(t_exp[0], 200)
        plt.ylim(0, psd_cutoff_min*10)
        axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.7f'))
        plt.title(f'Power Spectral Density after filtering (PSD cutoff = {psd_cutoff_min:0.9f})')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectrum |Fn|^2')
        plt.legend()

        plt.sca(axs[3])
        plt.plot(t_exp, signal_org, 'o', color='k', markersize=5, label='Noisy signal')
        plt.plot(t_exp, signal_clean, 'o', color='r', markersize=5, label='Filtered signal')
        # plt.plot(t_original, f_original, color='g', label='Original signal')
        plt.xlim(t_exp[0], t_exp[-1])
        plt.title('Filtered signal vs Original signal')
        plt.grid()
        plt.legend()

        fig.suptitle(f'Signal PSD Filtering with PSD cutoff = {psd_cutoff_min:0.9f}', fontsize=16)

        if return_plot:
            buf1 = io.BytesIO()
            plt.savefig(buf1, format='png', bbox_inches='tight',  dpi=45)
            plt.savefig(f'{logdir}/denoise_psd_graphs_{psd_cutoff_min:0.9f}.pdf', format='pdf', bbox_inches='tight',  dpi=100)
            buf1.seek(0)
            plt.show()
            buf1.close()
            return img, signal_org, signal_clean, p_s_d, p_s_d_clean
        else:
            plt.show()

    return img, signal_org, signal_clean, p_s_d, p_s_d_clean


def zero_pad_1D(img, pad_size):
    """
    Zero pad 1D array
    """
    img_pad = np.zeros((img.shape[0] + pad_size))
    img_pad[:img.shape[0]] = img
    return img_pad


def zero_pad_image(img, pad_size):
    """
    Zero pad image to the desired size
    """
    img_pad = np.zeros((img.shape[0], img.shape[1] + pad_size))
    img_pad[:, :img.shape[1]] = img
    return img_pad


all_results = []

for i in range(len(img_splits)):
    print(f'Processing image {i+1}/{len(img_splits)}')
    # for i in [7,8,9]:
    img_input = img_splits[i].copy()
    # img_input = zero_pad_image(img_input, pad_size=img_input.shape[1] * PADDING_MULTIPLIER)

    fft_results = fft_filter_img(img_input, PSD_CUTOFF, fft_bin_multiplier=10, plot=True, return_plot=True)

    img, signal_org, signal_clean, p_s_d, p_s_d_clean = fft_results
    # plt.figure(figsize=(15, 10))
    # plt.imshow(img, vmax=5000)

    all_results.append([signal_org, signal_clean, p_s_d, p_s_d_clean])

    for j in range(len(signal_org)):
        file_writer_1.add_scalars(main_tag='Signal',
                                  tag_scalar_dict={
                                      f'Original{i}': signal_org[j],
                                      f'Clean{i}': signal_clean[j],
                                  },
                                  global_step=j)
        file_writer_1.add_scalars(main_tag='PSD',
                                  tag_scalar_dict={
                                      f'Original{i}': p_s_d[j],
                                      f'Clean{i}': p_s_d_clean[j],
                                  },
                                  global_step=j)

# %%
for img_sp in img_split:
    plt.figure(figsize=(15, 10))
    plt.imshow(img_sp, vmax=5000)

# %%
all_results = np.array(all_results)

all_results[0][0].shape
