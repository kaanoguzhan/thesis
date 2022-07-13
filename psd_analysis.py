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
#  Image loading and splitting into sub-images
# ────────────────────────────────────────────────────────────────────────────────

# Load Image using AWAKE_DataLoader
adl = AWAKE_DataLoader('awake_1', [512, 672])
streak_img = adl.data[10].streak_image

# plot streak_img as a sanity check
plt.figure()
plt.imshow(streak_img, vmax=5000)
plt.savefig(f'{logdir}/0_streak_img.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
print(f'Streak image shape: {streak_img.shape}')

streak_img = streak_img.T

# Crop image from the beginning of Marker Laser Pulse
mlp_start, mlp_end = find_marker_laser_pulse(streak_img, x_axis_is_space=True)
print(f'Marker laser pulse is between: {mlp_start, mlp_end}')
streak_img = streak_img[:, 300:]
# streak_img = streak_img[:, mlp_end:]
# plot the streak_img after the Marker Laser Pulse cutting as a sanity check
plt.figure()
plt.imshow(streak_img, vmax=5000)
plt.savefig(f'{logdir}/1_streak_img_after_mlp_cut.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# Cut a window around the beam center and polot it as a sanity check
beam_window_start, beam_window_end = get_window_around_beam_centre(streak_img, args.beam_window, x_axis_is_space=True)
# beam_center = streak_img[290:300, :]
beam_center = streak_img[beam_window_start:beam_window_end, :]
plt.figure(figsize=(15, 10))
plt.imshow(beam_center, vmax=5000)
plt.savefig(f'{logdir}/2_streak_img_after_mlp_cut_beam_window.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

tmp_beam_center = copy.deepcopy(beam_center)

# Split image into sub-images with windows size of args.split_window_size and stride of args.split_stride
img_splits = split_image_with_stride(image=tmp_beam_center, window_size=args.split_window_size, stride=args.split_stride)

# Concat all images in img_split on top of eachother, leave 5 pixel wide zeros between each image
# Then plot it as a sanity check
img_cat = np.zeros(((img_splits[0].shape[0]+5) * len(img_splits) - 5, img_splits[0].shape[1]))
for idx, img_split in enumerate(img_splits):
    img_cat[idx*(img_splits[0].shape[0])+(idx)*5:(idx+1)*(img_splits[0].shape[0])+(idx)*5, :] = img_split
plt.figure(figsize=(15, 10))
plt.imshow(img_cat, vmax=5000)
plt.savefig(f'{logdir}/3_streak_img_after_mlp_cut_beam_window_split.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)


# %% ─────────────────────────────────────────────────────────────────────────────
#  Applying FFT filter and then doing the PSD analysis
# ────────────────────────────────────────────────────────────────────────────────

def fft_filter_img(image, psd_cutoff, fft_bin_multiplier, psd_frequency_range=(15, 200)):
    """
    Returns a fft filtered image.
    """
    dt_ps = 210 / 512  # Picoseconds (total time window:210 ps over 512 pixels)
    dt_fs = dt_ps * 1000  # Femtoseconds, this is to keep the precision of the frequency later on

    # Save image shape and row by sum on the first dimension
    img = image.copy()
    img_shape = img.shape
    signal = np.sum(img, axis=0)
    signal_org = signal.copy()
    signal_org = signal_org - np.mean(signal_org)  # detrend signal

    # Normalize signal between 0 and 1
    signal_max = np.max(signal)
    signal_min = np.min(signal)
    signal = (signal - signal_min) / (signal_max - signal_min)

    signal = signal - np.mean(signal)  # detrend signal

    t_exp = np.arange(len(signal), dtype=np.int32)

    # ─────────────────────────────────────────────────────────────
    #  Use PSD to filter out noise
    # ─────────────────────────────────────────────────────────────

    # Compute Fourier Coefficients of the sample signal
    Fn = np.fft.fft(signal, len(signal)*fft_bin_multiplier)

    # Compute Power Spectral Denisty
    p_s_d = (np.real(Fn)**2 + np.imag(Fn)**2) / len(signal)

    # p_s_d_phase = np.arctan(np.imag(Fn) / np.real(Fn))
    # p_s_d_phase = np.arctan2(np.imag(Fn), np.real(Fn)) * 180/np.pi
    # p_s_d_phase = np.arctan2(np.imag(Fn), np.real(Fn))

    X2=Fn.copy() 
    threshold = np.max(np.abs(X2))/100
    X2[np.abs(X2)<threshold] = 0
    p_s_d_phase = np.arctan2(np.imag(X2), np.real(X2)) * 180/np.pi
    # p_s_d_phase = np.angle(Fn) * 180/np.pi

    # # Normalize PSD
    # p_s_d_min = np.min(p_s_d)
    # p_s_d_max = np.max(p_s_d)
    # p_s_d = (p_s_d - p_s_d_min) / (p_s_d_max - p_s_d_min)

    # Apply PSD cutoff
    if isinstance(psd_cutoff, tuple):
        psd_cutoff_max = psd_cutoff[0]
        psd_cutoff_min = psd_cutoff[1]
    else:
        psd_cutoff_max = np.inf
        psd_cutoff_min = psd_cutoff
    indices = np.logical_and(p_s_d > psd_cutoff_min, p_s_d < psd_cutoff_max)

    # Zero out smaller Fourier coefficients and their corresponding frequencies
    p_s_d_clean = p_s_d * indices

    # Go over each row on image and replace with PSD filtered signal
    for i in range(img_shape[0]):
        signal_row = img[i]
        signal_row = signal_row - np.mean(signal_row)
        Fn_row = np.fft.fft(signal_row, len(signal_row)*fft_bin_multiplier)
        Fn_row = Fn_row * indices
        img_row_ifft = np.fft.ifft(Fn_row)
        img[i] = np.real(img_row_ifft[:img_shape[1]])

    signal_clean = np.sum(img, axis=0)

    ghz_freqs = np.power(dt_fs, -1) * 1000 * 1000  # dt_fs^-1 *1000*1000 = PetaHz *1000*1000 => GigaHz
    freq_ticks = np.fft.fftfreq(len(signal)*fft_bin_multiplier) * ghz_freqs

    # Set all frequencies that are not in psd_frequency_range to None
    frequency_range_start = np.where(freq_ticks > psd_frequency_range[0])[0][0]
    frequency_range_end = np.where(freq_ticks > psd_frequency_range[1])[0][0]
    p_s_d[:frequency_range_start] = None
    p_s_d[frequency_range_end:] = None
    p_s_d_clean[:frequency_range_start] = None
    p_s_d_clean[frequency_range_end:] = None
    p_s_d_phase[:frequency_range_start] = None
    p_s_d_phase[frequency_range_end:] = None

    # fig, ax = plt.subplots()
    # ax.plot(freq_ticks, p_s_d)
    # ax.plot(freq_ticks, p_s_d, 'o', color='tab:blue', markersize=3)
    # ax.set_xlabel('Frequency [GHz]')
    # ax.set_ylabel('Power')
    # # plt.xticks(range(0, psd_frequency_range[-1]+10, 10))
    # # plt.xlim()
    # ax.set_xlim(0, psd_frequency_range[-1]+20)
    # ax.set_ylim(0, 1.05)
    # ax.set_xticks(range(0, 250, 10))
    # ax.grid()
    # fig.savefig(f'plot_PSD.pdf')

    # Calculate Time and Frequency Ticks
    time_ticks = np.arange(0, t_exp[-1], 1) * dt_fs
    time_ticks = [int(i) for i in time_ticks]  # in Femtoseconds

    ghz_freqs = np.power(dt_fs, -1) * 1000 * 1000  # dt_fs^-1 *1000*1000 = PetaHz *1000*1000 => GigaHz
    freq_ticks = np.fft.fftfreq(len(signal)*fft_bin_multiplier) * ghz_freqs

    # Plotting denoising results
    fig, axs = plt.subplots(5, 1, figsize=(20, 22))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.2)

    plt.sca(axs[0])
    plt.plot(t_exp, signal_org, color='k')
    plt.plot(t_exp, signal_org, 'o', color='k', label='Noisy signal')
    plt.xlim(t_exp[0], t_exp[-1])
    plt.xticks(range(0, len(time_ticks), 10), time_ticks[::10])
    plt.ylim(np.min(signal_org), np.max(signal_org))
    plt.title('Noisy signal (mean removed)')
    plt.xlabel('Time [fs]')
    plt.ylabel('Pixel Intensity (Column Sum)')
    plt.grid()
    plt.legend()

    plt.sca(axs[1])
    plt.plot(freq_ticks, p_s_d, color='b')
    plt.plot(freq_ticks, p_s_d, 'o', color='b', linewidth=2, label='Noisy signal')
    plt.plot([psd_frequency_range[0], psd_frequency_range[-1]], [psd_cutoff_min, psd_cutoff_min], '--', color='tab:orange', label='PSD cutoff')
    plt.plot([psd_frequency_range[0], psd_frequency_range[-1]], [psd_cutoff_max, psd_cutoff_max], '--', color='tab:orange', label='PSD cutoff')
    plt.xticks(range(0, psd_frequency_range[-1]+10, 10))
    plt.xlim(0, psd_frequency_range[-1])
    plt.ylim(0, 2)
    plt.title(f'Power Spectral Density')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Power')
    plt.grid()
    plt.legend()

    plt.sca(axs[2])
    plt.plot(freq_ticks, p_s_d, color='b')
    plt.plot(freq_ticks, p_s_d_clean, color='darkred')
    plt.plot(freq_ticks, p_s_d, 'o', color='b', linewidth=2, label='Noisy signal')
    plt.plot(freq_ticks, p_s_d_clean, 'o', color='darkred', linewidth=2, label='Filtered signal')
    plt.plot([psd_frequency_range[0], psd_frequency_range[-1]], [psd_cutoff_min, psd_cutoff_min], '--', color='tab:orange', label='PSD cutoff')
    plt.plot([psd_frequency_range[0], psd_frequency_range[-1]], [psd_cutoff_max, psd_cutoff_max], '--', color='tab:orange', label='PSD cutoff')
    plt.xticks(range(0, psd_frequency_range[-1]+10, 10))
    plt.xlim(0, psd_frequency_range[-1])
    plt.ylim(0, 2)
    plt.title(f'Power Spectral Density after filtering (PSD cutoff = {psd_cutoff_min:0.9f})')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Power')
    plt.grid()
    plt.legend()
    
    plt.sca(axs[3])
    plt.plot(freq_ticks, p_s_d_phase, color='b')
    plt.plot(freq_ticks, p_s_d_phase, 'o', color='b', linewidth=2, label='Phase')
    plt.xticks(range(0, psd_frequency_range[-1]+10, 10))
    plt.xlim(0, psd_frequency_range[-1])
    # plt.ylim(0, 2)
    plt.title(f'Phase')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Phase')
    plt.grid()
    plt.legend()

    plt.sca(axs[4])
    plt.plot(t_exp, signal_org, 'o', color='k', markersize=5, label='Noisy signal')
    plt.plot(t_exp, signal_clean, 'o', color='r', markersize=5, label='Filtered signal')
    plt.xlim(t_exp[0], t_exp[-1])
    plt.xticks(range(0, len(time_ticks), 10), time_ticks[::10])
    plt.title('Filtered signal vs Original signal (mean removed)')
    plt.xlabel('Time [fs]')
    plt.ylabel('Pixel Intensity (Column Sum)')
    plt.grid()
    plt.legend()

    fig.suptitle(f'Signal PSD Filtering with PSD cutoff ({psd_cutoff_min:0.4f},{psd_cutoff_max:0.4f})', fontsize=16)

    return img, time_ticks, signal_org, signal_clean, freq_ticks, p_s_d, p_s_d_clean, p_s_d_phase, plt


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
    img_input = img_splits[i].copy()

    fft_results = fft_filter_img(img_input, args.psd_cutoff, fft_bin_multiplier=args.padding_multiplier)
    img, time_ticks, signal_org, signal_clean, freq_ticks, p_s_d, p_s_d_clean, p_s_d_phase, fft_signal_plot = fft_results

    fft_signal_plot.savefig(f'{logdir}/denoise_psd_graphs_{i}.pdf', bbox_inches='tight')

    all_results.append([
        np.array(time_ticks),
        np.array(signal_org),
        np.array(signal_clean),
        np.array(freq_ticks),
        np.array(p_s_d),
        np.array(p_s_d_clean),
        np.array(p_s_d_phase)
    ])

    # TODO saving results to tensorboard
    # for j in range(len(time_ticks)):
    #     file_writer_1.add_scalars(main_tag='Signal',
    #                               tag_scalar_dict={
    #                                   f'Original{i}': signal_org[j],
    #                                   f'Clean{i}': signal_clean[j],
    #                               },
    #                               global_step=time_ticks[j])
    # for j in range(len(freq_ticks)):
    #     file_writer_1.add_scalars(main_tag='PSD',
    #                               tag_scalar_dict={
    #                                   f'Original{i}': p_s_d[j],
    #                                   f'Clean{i}': p_s_d_clean[j],
    #                               },
    #                               global_step=freq_ticks[j])

# %% ─────────────────────────────────────────────────────────────────────────────
#  Merge denoise_psd_graphs_N.pdf files into single pdf
# ────────────────────────────────────────────────────────────────────────────────

# Read all PDF files under logs/ directory
pdf_files = [f for f in os.listdir(logdir) if f.endswith('.pdf')]
pdf_files = natural_sort(pdf_files)
pdf_files = [os.path.join(logdir, f) for f in pdf_files]

# Categorize PDF files
denoise_psd_graphs = [f for f in pdf_files if 'denoise_psd_graphs_' in f]

# Merge PDF files
merge_pdfs(denoise_psd_graphs, os.path.join(logdir, '4_denoise_psd_graphs.pdf'))

# Delete all residual PDF files after merging
for f in pdf_files:
    if 'denoise_psd_graphs_' in f:
        os.remove(f)


# %% ─────────────────────────────────────────────────────────────────────────────
#  Plotting Clean signal and their PSD for each image into a single figure
# ────────────────────────────────────────────────────────────────────────────────
all_results_copy = copy.deepcopy(all_results)
all_results_copy = np.array(all_results_copy)

sum_of_signal_clean = np.sum(all_results_copy[:, 2], axis=0)
sum_of_signal_clean.shape

color = np.random.rand(10000, 3,)
signal_clean = all_results_copy[:, 2][0]
t_exp = np.arange(len(signal_clean), dtype=np.int32)
freq_ticks = all_results_copy[:, 3][0]

# plot the sum of the signal
xlim = (0, 100)
plt.figure(figsize=(15, 10))
for idx, clean in enumerate(all_results_copy[:, 2]):
    plt.plot(t_exp, clean, 'o', markersize=4, color=color[idx], label=f'Clean-{idx}')
    plt.plot(t_exp, clean, color=color[idx])
plt.xlim(xlim)
plt.xticks(np.arange(0, xlim[1]+1, 5))
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(f'Clean signals | PAD:{args.padding_multiplier} | PSD_CUTOFF:{args.psd_cutoff}')
plt.savefig(f'{logdir}/5_clean_signals_combined.pdf', format='pdf', bbox_inches='tight')

# plot the sum p_s_d_clean
xlim = (0, 200)
plt.figure(figsize=(15, 10))
for idx, clean in enumerate(all_results_copy[:, 5]):
    plt.plot(freq_ticks, clean, 'o', markersize=4, color=color[idx], label=f'Clean-{idx}')
    plt.plot(freq_ticks, clean, color=color[idx])
plt.xlim(xlim)
plt.xticks(np.arange(0, xlim[1]+1, 5))
plt.ylim(0, 2)
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(f'Clean PSD | Zero-Pad:{args.padding_multiplier} | PSD_CUTOFF:{args.psd_cutoff}')
plt.savefig(f'{logdir}/5_clean_PSDs_combined.pdf', format='pdf', bbox_inches='tight')

# %% ─────────────────────────────────────────────────────────────────────────────
#  Histogram plotting of signal peaks from the clean
# ────────────────────────────────────────────────────────────────────────────────
all_results_copy = copy.deepcopy(all_results)
all_results_copy = np.array(all_results_copy)

p_s_d_clean = all_results_copy[:, 5].copy()
p_s_d_phase = all_results_copy[:, 6].copy()
freq_ticks = all_results_copy[:, 3].copy()

# Project high-resolution PSD to low-resolution(integer) PSD
psd_low_res = np.zeros(200)
psd_peak_freqs = np.array([])
psd_peak_phases = np.array([])
for idx, clean in enumerate(p_s_d_clean):
    # replace nan with 0
    clean[np.isnan(clean)] = 0

    # Find the peak
    clean_max = np.argmax(clean)
    print(f'Image-{idx:>02} Peak Frequency: {freq_ticks[idx][clean_max]:.2f}')

    # Make all values of p_s_d_clean 0 except the peak
    clean[:] = 0
    clean[clean_max] = 1

    peak_freq = freq_ticks[idx][clean_max]
    peak_phase = p_s_d_phase[idx][clean_max]
    psd_peak_freqs = np.append(psd_peak_freqs, peak_freq)
    psd_peak_phases = np.append(psd_peak_phases, peak_phase)
    psd_low_res[int(peak_freq)] += 1

# High Res Frequency Plot (Keeps all frequencies as floating numbers)
# psd_sum = np.sum(p_s_d_clean, axis=0)
# xlim = (0, 200)
# plt.figure(figsize=(15, 10))
# plt.bar(freq_ticks[0], psd_sum, color='red')
# plt.xlim(xlim)
# plt.xticks(np.arange(0, xlim[1]+1, 5))
# plt.yticks(np.arange(0, np.max(psd_sum)+1, 1))
# plt.grid()
# plt.suptitle(f'Peak Histogram | BEAM_WINDOW:{BEAM_WINDOW} - SPLIT_SIZE: {SPLIT_WINDOW_SIZE} - STRIDE: {SPLIT_STRIDE}', fontsize=16)
# plt.savefig(f'{logdir}/6_clean_psd_histogram_high_res.pdf', bbox_inches='tight')

# Low Res Frequcy Plot (reduces all frequencies to integers)
xlim = (0, 200)
plt.figure(figsize=(15, 10))
plt.bar(np.arange(len(psd_low_res)), psd_low_res, color='red')
plt.xlim(xlim)
plt.xticks(np.arange(0, xlim[1]+1, 5))
plt.yticks(np.arange(0, np.max(psd_low_res)+1, 1))
plt.grid()
plt.suptitle(f'Peak Histogram | Samples:{len(all_results_copy)} | BEAM_WINDOW:{args.beam_window} - SPLIT_SIZE: {args.split_window_size} - STRIDE: {args.split_stride}', fontsize=16)
plt.savefig(f'{logdir}/6_clean_psd_histogram_low_res.pdf', bbox_inches='tight')

# %% ─────────────────────────────────────────────────────────────────────────────
#  Mark sub images that have a frequency peak between 85 and 100
# ────────────────────────────────────────────────────────────────────────────────
# Concat all images in img_split on top of eachother, leave 5 pixel wide zeros between each image
split_height = img_splits[0].shape[0]
img_cat = np.zeros(((split_height+5) * len(img_splits) + 5, img_splits[0].shape[1]))
for idx, img_split in enumerate(img_splits):
    img_cat[idx*(split_height)+(idx)*5+5:(idx+1)*(split_height)+(idx)*5+5, :] = img_split

# Get indexes of psd_peaks where the frequency is between 85 - 100
idx_peaks_between = np.where((psd_peak_freqs > 85) & (psd_peak_freqs < 100))[0]
idx_peaks_outside = np.where((psd_peak_freqs < 85) | (psd_peak_freqs > 100))[0]
peak_index_not_between = np.where((psd_peak_freqs < 85) | (psd_peak_freqs > 100))[0]
for idx in idx_peaks_between:
    img_cat[idx*(split_height)+(idx)*5+3:(idx)*(split_height)+(idx)*5+5, :] = 4000
    img_cat[(idx+1)*(split_height)+(idx)*5+5:(idx+1)*(split_height)+(idx)*5+7, :] = 4000
for idx in peak_index_not_between:
    img_cat[idx*(split_height)+(idx)*5+3:(idx)*(split_height)+(idx)*5+5, :] = 2000
    img_cat[(idx+1)*(split_height)+(idx)*5+5:(idx+1)*(split_height)+(idx)*5+7, :] = 2000
plt.figure(figsize=(15, 10))
plt.imshow(img_cat, vmax=5000, interpolation='none')
plt.savefig(f'{logdir}/7_marked_splits.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# %% ─────────────────────────────────────────────────────────────────────────────
#  Calculating Mean & Variance of the peaks
# ────────────────────────────────────────────────────────────────────────────────
psd_peak_freqs_between = psd_peak_freqs[idx_peaks_between]
psd_peak_phases_between = psd_peak_phases.copy()
psd_peak_phases_between = psd_peak_phases_between[idx_peaks_between]

# Calculate mean and std of the peaks and variance
mean_peak = np.mean(psd_peak_freqs_between)
var_peak = np.var(psd_peak_freqs_between)

print(f'Mean Peak: {mean_peak:.2f} | Var Peak: {var_peak:.4f}')

# Sanity check
# Merge all image splits that have a frequency peak between 85 and 100
img_cat = img_splits[idx_peaks_between]
img_cat = np.concatenate(img_cat, axis=0)
# Apply the fft_filter to the clean psd
a = fft_filter_img(img_cat, args.psd_cutoff, fft_bin_multiplier=args.padding_multiplier)
_, _, _, _, freq_ticks, p_s_d, _, _, _ = a
# Find the peak
p_s_d[np.isnan(p_s_d)] = 0
idx_peak = np.argmax(p_s_d)
max_freq = freq_ticks[idx_peak]
print(f'Max Frequency: {max_freq:.2f}')
if np.abs(max_freq - mean_peak) < 1.5:
    print('!!! Max Frequency diverges too much from the peak mean !!!')

# Write results to JSON file
result = {
    'peak_between_85_100_ghz': {
        'mean': mean_peak,
        'var': var_peak
    },
    'combined_image_max_freq': max_freq
}
with open(f'{logdir}/signal_peak_analysis.json', 'w') as f:
    json.dump(result, f, indent=4)

# Calcualte the phase of the signal at the peak frequency
# Get the phase of the signal at the peak frequency
phase_at_peak = np.abs(psd_peak_phases_between)
print(f'Phase at Peak: {phase_at_peak}')

# plot the phase of the signal at the peak frequency as histogram
plt.figure(figsize=(15, 10))
plt.hist(phase_at_peak, bins=30)
plt.grid()
plt.suptitle(f'Phase at Frequency Peaks | Samples:{len(all_results_copy)} | BEAM_WINDOW:{args.beam_window} - SPLIT_SIZE: {args.split_window_size} - STRIDE: {args.split_stride}', fontsize=16)
plt.savefig(f'{logdir}/8_phase_at_peaks.pdf', bbox_inches='tight')


# %% ─────────────────────────────────────────────────────────────────────────────
#  Plot Freq Peak vs Beam window number
# ────────────────────────────────────────────────────────────────────────────────

# Split into between 85 and 100 and outside
psd_peak_freqs_between = psd_peak_freqs[idx_peaks_between]
psd_peak_freqs_outside = psd_peak_freqs[idx_peaks_outside]

# make limited copies with max 120 Ghz for plotting
psd_peak_freqs_between_lim = copy.deepcopy(psd_peak_freqs_between)
psd_peak_freqs_outside_lim = copy.deepcopy(psd_peak_freqs_outside)
psd_peak_freqs_between_lim[psd_peak_freqs_between_lim > 120] = 120
psd_peak_freqs_outside_lim[psd_peak_freqs_outside_lim > 120] = 120

plt.figure(figsize=(15, 10))
plt.plot(idx_peaks_between, psd_peak_freqs_between_lim, color='mediumseagreen', marker='o', markersize=10, label='Peak Frequency Between 85 and 100 GHz')
plt.plot(idx_peaks_outside, psd_peak_freqs_outside_lim, 'o', color='tomato', markersize=10, label='Peak Frequency Outside 85 and 100 GHz')
for x, y, t in zip(idx_peaks_between, psd_peak_freqs_between_lim, psd_peak_freqs_between):
    plt.text(x, y, f'{t:.1f}', ha='center', va='bottom')
for x, y, t in zip(idx_peaks_outside, psd_peak_freqs_outside_lim, psd_peak_freqs_outside):
    plt.text(x, y, f'{t:.1f}', ha='center', va='bottom')
if len(psd_peak_freqs) < 30:
    plt.xticks(range(len(psd_peak_freqs)))
elif len(psd_peak_freqs) < 60:
    plt.xticks(range(0, len(psd_peak_freqs)+2, 2))
else:
    plt.xticks(range(0, len(psd_peak_freqs)+3, 3))
plt.xlim(-1, len(psd_peak_freqs))
plt.yticks(np.arange(0, 121, 10))
plt.ylim(0, 120)
plt.grid(axis='y')
plt.xlabel('Beam Window Number')
plt.ylabel('Frequency (GHz)')
plt.suptitle(f'Frequency Peaks for each Beam Window | Samples:{len(all_results_copy)} | BEAM_WINDOW:{args.beam_window} - SPLIT_SIZE: {args.split_window_size} - STRIDE: {args.split_stride}', fontsize=16)
plt.legend()
plt.savefig(f'{logdir}/9_freq_at_peaks_vs_beam_window.pdf', bbox_inches='tight')


# %% ─────────────────────────────────────────────────────────────────────────────
#  Plot Phase at Frequency Peak vs Beam window number
# ────────────────────────────────────────────────────────────────────────────────

# Split into between 85 and 100 and outside
psd_peak_phases_between = psd_peak_phases[idx_peaks_between]
psd_peak_phases_outside = psd_peak_phases[idx_peaks_outside]

# make limited copies with between -120, 120 Degree for plotting
psd_peak_phases_between_lim = copy.deepcopy(psd_peak_phases_between)
psd_peak_phases_outside_lim = copy.deepcopy(psd_peak_phases_outside)
psd_peak_phases_between_lim[psd_peak_phases_between_lim > 120] = 120
psd_peak_phases_outside_lim[psd_peak_phases_outside_lim > 120] = 120
psd_peak_phases_between_lim[psd_peak_phases_between_lim < -120] = -120
psd_peak_phases_outside_lim[psd_peak_phases_outside_lim < -120] = -120

plt.figure(figsize=(15, 10))
plt.plot(idx_peaks_between, psd_peak_phases_between_lim, color='mediumseagreen', marker='o', markersize=10, label='Phase at Frequency Peak Between 85 and 100 GHz')
plt.plot(idx_peaks_outside, psd_peak_phases_outside_lim, 'o', color='tomato', markersize=10, label='Phase at Frequency Peak Outside 85 and 100 GHz')
# write value on top of each bar
for x, y, t in zip(idx_peaks_between, psd_peak_phases_between_lim, psd_peak_phases_between):
    plt.text(x, y, f'{t:.1f}', ha='center', va='bottom')
for x, y, t in zip(idx_peaks_outside, psd_peak_phases_outside_lim, psd_peak_phases_outside):
    plt.text(x, y, f'{t:.1f}', ha='center', va='bottom')
if len(psd_peak_freqs) < 30:
    plt.xticks(range(len(psd_peak_freqs)))
elif len(psd_peak_freqs) < 60:
    plt.xticks(range(0, len(psd_peak_freqs)+2, 2))
else:
    plt.xticks(range(0, len(psd_peak_freqs)+3, 3))
plt.xlim(-1, len(psd_peak_freqs))
plt.yticks(np.arange(-120, 121, 30))
plt.ylim(-120, 120)
plt.grid(axis='y')
plt.xlabel('Beam Window Number')
plt.ylabel('Phase (deg)')
plt.suptitle(f'Phase at Frequency Peaks for each Beam Window | Samples:{len(all_results_copy)} | BEAM_WINDOW:{args.beam_window} - SPLIT_SIZE: {args.split_window_size} - STRIDE: {args.split_stride}', fontsize=16)
plt.legend()
plt.savefig(f'{logdir}/9_phase_at_peaks_vs_beam_window.pdf', bbox_inches='tight')


# %% ─────────────────────────────────────────────────────────────────────────────
#  Frequency vs Phase
# ────────────────────────────────────────────────────────────────────────────────

psd_peak_freqs_between = psd_peak_freqs[idx_peaks_between]
psd_peak_phases_between = psd_peak_phases[idx_peaks_between]

# make limited copies with between -120, 120 Degree for plotting
psd_peak_freqs_between_lim = copy.deepcopy(psd_peak_freqs_between)
psd_peak_phases_between_lim = copy.deepcopy(psd_peak_phases_between)
psd_peak_freqs_between_lim[psd_peak_freqs_between_lim > 120] = 120
psd_peak_phases_between_lim[psd_peak_phases_between_lim > 120] = 120
psd_peak_phases_between_lim[psd_peak_phases_between_lim < -120] = -120

# plot psd_peak_freqs in bar graph
plt.figure(figsize=(15, 10))
plt.plot(psd_peak_freqs_between, psd_peak_phases_between, color='mediumseagreen', marker='o', markersize=10, label='Peak Frequency Between 85 and 100 GHz')
# write value on top of each bar
for i, x, y, t in zip(range(len(psd_peak_freqs_between)), psd_peak_freqs_between, psd_peak_phases_between_lim, psd_peak_phases_between):
    plt.text(x, y, f'Win{i}', ha='center', va='bottom')
plt.ylim(-120, 120)
plt.yticks(np.arange(-120, 121, 30))
plt.xticks(np.arange(85, 101, 1))
plt.grid()
plt.xlabel('Frequency (GHz)')
plt.ylabel('Phase (deg)')
plt.suptitle(f'Frequency vs Phase at Frequency Peaks | Samples:{len(all_results_copy)} | BEAM_WINDOW:{args.beam_window} - SPLIT_SIZE: {args.split_window_size} - STRIDE: {args.split_stride}', fontsize=16)
plt.legend()
plt.savefig(f'{logdir}/9_freq_at_peaks_vs_phase_at_peaks.pdf', bbox_inches='tight')


