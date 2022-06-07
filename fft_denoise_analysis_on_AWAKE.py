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
import tensorflow as tf
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.awake_data_loader import AWAKE_DataLoader
from utils.beam_utils import get_window_around_beam_center
from utils.general_utils import merge_pdfs, natural_sort

ENABLE_RUNNING_BY_JUPYTER_NOTEBOOK = False  # Set to True enable running from Jupyter Notebooks

if ENABLE_RUNNING_BY_JUPYTER_NOTEBOOK:
    sys.argv = ['']
else:
    matplotlib.use('Agg')

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_BEAM_WINDOW = 10
DEFAULT_NUM_PSD_STEPS = 70

adl = AWAKE_DataLoader('awake_1', [512, 672])

cur_data = adl.data[0]

# Initilize arg parser
parser = argparse.ArgumentParser(description='FFT denoise analysis')
parser.add_help = True
parser.add_argument('--beam_window', type=int, default=DEFAULT_BEAM_WINDOW, help='Beam Windows width around the beam center')
parser.add_argument('--num_psd_steps', type=int, default=DEFAULT_NUM_PSD_STEPS, help='Number of PSD steps in np.logspace(-2, -6, args.num_psd_steps)')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.figsize': [15, 10],
    'savefig.transparent': False,
    'savefig.facecolor': 'w'
})


# Sets up log directory with timestamp
logdir = "logs/" + datetime.now().strftime("%Y.%m.%d-%H_%M_%S") + f'_AWAKE_FFT_Denoising-window{args.beam_window}'
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

print(f'\
    Parameters:\n\
    ----------------------------------------\n\
    Beam window: {args.beam_window}\n\
    Number of PSD steps: {args.num_psd_steps}\n\
    Log directory: {logdir}\n\
    ----------------------------------------\n\
    ')


# %% ─────────────────────────────────────────────────────────────────────────────
# Function definitions
# ────────────────────────────────────────────────────────────────────────────────


def fft_filter(signal, psd_cutoff, plot_steps=False):
    """
    signal: 1D Signal (2D numpy array)
    psd_cutoff: Cutoff frequency in PSD (float)

    Returns: FFT filtered signal of the given signal.
    """

    t_experiment = np.arange(len(signal))

    # Normalize signal between 0 and 1
    signal_old = signal.copy()
    signal_max = np.max(signal)
    signal = signal / signal_max

    # Compute Fourier Coefficients of the signal
    # Fourier Coefficients have magnitude and phase
    Fn = np.fft.fft(signal, len(signal))

    # -------------------------------------------------------------------------
    # Use PSD to filter out noise

    # Compute Power Spectral Denisty
    p_s_d = Fn * np.conj(Fn) / len(signal)

    # Find all frequencies with large power
    indices = p_s_d > psd_cutoff

    # Zero out smaller Fourier coefficients and their corresponding frequencies
    Fn = Fn * indices
    PSDClean = p_s_d * indices

    # Apply Inverse FFT
    ffilt = np.fft.ifft(Fn)

    # Reverse normalization
    ffilt *= signal_max

    freq = np.arange(len(signal)/2, dtype=np.int32)

    if plot_steps:
        # Plotting denoising results
        fig, axs = plt.subplots(4, 1, figsize=(20, 22))
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.3)

        plt.sca(axs[0])
        plt.plot(t_experiment, signal, color='k')
        plt.plot(t_experiment, signal, 'o', color='k', label='Noisy signal')
        plt.xlim(t_experiment[0], t_experiment[-1])
        plt.title('Noisy signal')
        plt.grid()
        plt.legend()

        plt.sca(axs[1])
        plt.plot(freq, p_s_d[freq], 'o', color='b', linewidth=2, label='Noisy signal')
        plt.plot(freq, p_s_d[freq], color='b')
        plt.plot([freq[0], freq[-1]], [psd_cutoff, psd_cutoff], '--', color='tab:orange', label='PSD cutoff')
        plt.xlim(freq[0], freq[-1])
        plt.title(f'Power Spectral Density before filtering')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectrum |Fn|^2')
        plt.legend()

        plt.sca(axs[2])
        plt.plot(freq, PSDClean[freq], color='darkred')
        plt.plot(freq, PSDClean[freq], 'o', color='darkred', linewidth=2, label='Filtered signal')
        plt.plot([freq[0], freq[-1]], [psd_cutoff, psd_cutoff], '--', color='tab:orange', label='PSD cutoff')
        plt.xlim(freq[0], freq[-1])
        plt.title(f'Power Spectral Density after filtering')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectrum |Fn|^2')
        plt.legend()

        plt.sca(axs[3])
        plt.plot(t_experiment, signal_old, 'o', color='k', markersize=10, label='Noisy signal')
        plt.plot(t_experiment, ffilt, 'o', color='r', markersize=5, label='Filtered signal')
        plt.xlim(t_experiment[0], t_experiment[-1])
        plt.title('Filtered signal vs Original signal')
        plt.grid()
        plt.legend()

        plt.show()

    return ffilt


def fft_filter_img(img, psd_cutoff, plot=False, return_plot=False):
    """
    Returns a fft filtered image.
    """

    # Save image shape and row by sum on the first dimension
    img_shape = img.shape
    signal = np.sum(img, axis=0)

    # Normalize signal between 0 and 1
    signal_old = signal.copy()
    signal_max = np.max(signal)
    signal = signal / signal_max

    t_experiment = np.arange(len(signal))

    # Compute Fourier Coefficients of the f_sample signal
    # Fourier Coefficients have magnitude and phase
    Fn = np.fft.fft(signal, len(signal))

    # Use PSD to filter out noise

    # Compute Power Spectral Denisty
    p_s_d = Fn * np.conj(Fn) / len(signal)

    # Normalize PSD
    p_s_d_max = np.max(p_s_d)
    p_s_d /= p_s_d_max

    # Find all frequencies with large power
    indices = p_s_d > psd_cutoff

    # Zero out smaller Fourier coefficients and their corresponding frequencies
    # Boradcast Fn to the same shape as img_shape
    Fn = Fn * indices
    PSDClean = p_s_d * indices

    # Apply Inverse FFT
    ffilt = np.fft.ifft(Fn)

    # Reverse normalization
    ffilt *= signal_max

    # Print Magnitude and Phase of the filtered signal
    # print(f'Magnitude of the filtered signal: {np.abs(ffilt)}')
    # print(f'Phase of the filtered signal: {np.angle(ffilt)}')

    # Go over each row on image and replace with filtered signal
    for i in range(img_shape[0]):
        signal_row = img[i]
        Fn_row = np.fft.fft(signal_row, len(signal_row))
        Fn_row = Fn_row * indices
        img_ifft = np.fft.ifft(Fn_row)
        img[i] = np.real(img_ifft)

    freq = np.arange(len(signal)/2, dtype=np.int32)

    if plot:
        # Clean p_s_d and PSDClean from imaginary part
        PSDClean = np.real(PSDClean)
        p_s_d = np.real(p_s_d)

        # Plotting denoising results
        fig, axs = plt.subplots(4, 1, figsize=(20, 22))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.2)

        plt.sca(axs[0])
        plt.plot(t_experiment, signal, color='k')
        plt.plot(t_experiment, signal, 'o', color='k', label='Noisy signal')
        plt.xlim(t_experiment[0], t_experiment[-1])
        plt.title('Noisy signal')
        plt.ylim(0, 1)
        plt.grid()
        plt.legend()

        plt.sca(axs[1])
        plt.plot(freq, p_s_d[freq], color='b')
        plt.plot(freq, p_s_d[freq], 'o', color='b', linewidth=2, label='Noisy signal')
        plt.plot([freq[0], freq[-1]], [psd_cutoff, psd_cutoff], '--', color='tab:orange', label='PSD cutoff')
        plt.xlim(freq[0], freq[-1])
        # plt.ylim(0, psd_cutoff*10)
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        plt.title(f'Power Spectral Density')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectrum |Fn|^2')
        plt.legend()

        plt.sca(axs[2])
        plt.plot(freq, p_s_d[freq], color='b')
        plt.plot(freq, PSDClean[freq], color='darkred')
        plt.plot(freq, p_s_d[freq], 'o', color='b', linewidth=2, label='Noisy signal')
        plt.plot(freq, PSDClean[freq], 'o', color='darkred', linewidth=2, label='Filtered signal')
        plt.plot([freq[0], freq[-1]], [psd_cutoff, psd_cutoff], '--', color='tab:orange', label='PSD cutoff')
        plt.xlim(freq[0], freq[-1])
        plt.ylim(0, psd_cutoff*10)
        axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.7f'))
        plt.title(f'Power Spectral Density after filtering (PSD cutoff = {psd_cutoff:0.9f})')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectrum |Fn|^2')
        plt.legend()

        plt.sca(axs[3])
        plt.plot(t_experiment, signal_old, 'o', color='k', markersize=5, label='Noisy signal')
        plt.plot(t_experiment, np.sum(img, axis=0), 'o', color='r', markersize=5, label='Filtered signal')
        # plt.plot(t_original, f_original, color='g', label='Original signal')
        plt.xlim(t_experiment[0], t_experiment[-1])
        plt.title('Filtered signal vs Original signal')
        plt.grid()
        plt.legend()

        fig.suptitle(f'Signal PSD Filtering with PSD cutoff = {psd_cutoff:0.9f}', fontsize=16)

        if return_plot:
            buf1 = io.BytesIO()
            plt.savefig(buf1, format='png', bbox_inches='tight',  dpi=45)
            plt.savefig(f'{logdir}/denoise_psd_graphs_{psd_cutoff:0.9f}.pdf', format='pdf', bbox_inches='tight',  dpi=100)
            buf1.seek(0)
            # Convert PNG buffer to TF image
            plt_image = tf.image.decode_png(buf1.getvalue(), channels=4)
            # Add the batch dimension
            plt_image = tf.expand_dims(plt_image, 0)
            plt.show()
            buf1.close()
            return img, plt_image
        else:
            plt.show()

    return img, None


# %% ─────────────────────────────────────────────────────────────────────────────
# Denoise the image using the FFT-PSD method
# ────────────────────────────────────────────────────────────────────────────────
cur_img = cur_data.streak_image[265:]

cur_img = cur_img.T

temp_img = cur_img

temp_img = get_window_around_beam_center(temp_img, args.beam_window)

tif = temp_img.copy()

for idx, psd_cutoff in enumerate(np.logspace(-2, -6, args.num_psd_steps)[::-1]):
    tif, filter_plot = fft_filter_img(img=temp_img.copy(), psd_cutoff=psd_cutoff, plot=True, return_plot=True)

    fig, axs = plt.subplots(4, 1, figsize=(20, 16))
    # fig.tight_layout(rect=[0, 0.03, 1, 1])
    plt.subplots_adjust(hspace=0.2)

    plt.sca(axs[0])
    im = plt.imshow(temp_img, vmax=5000)
    plt.title('Cropped Image')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.sca(axs[1])
    im = plt.imshow(tif, vmax=5000)
    plt.title('Denoised Image')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    diff = np.abs(tif - temp_img)
    plt.sca(axs[2])
    plt.imshow(diff, cmap='Reds', vmax=4500)
    plt.imshow(tif, alpha=0.6)
    plt.title('Difference (Denoised Image + Heatmap)')
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    zeros = np.ones([100, 10])
    plt.imshow(zeros, vmax=1, vmin=0, cmap='gray')
    # Hide everything for the colorbar
    cax.xaxis.set_ticks_position('none')
    cax.yaxis.set_ticks_position('none')
    plt.setp(cax.get_xticklabels(), visible=False)
    plt.setp(cax.get_yticklabels(), visible=False)
    cax.spines['bottom'].set_color('white')
    cax.spines['top'].set_color('white')
    cax.spines['left'].set_color('white')
    cax.spines['right'].set_color('white')

    plt.sca(axs[3])
    im = plt.imshow(diff, cmap='Reds', vmax=4500)
    plt.title('Difference (Heatmap)')
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    fig.suptitle(f'Image Comparison of Denoised Image with "psd_cutoff={psd_cutoff:0.9f}"', fontsize=16)

    plt.savefig(f'{logdir}/image_comparison_{psd_cutoff:0.9f}.pdf', format='pdf', bbox_inches='tight',  dpi=100)

    # Save figure to Tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',  dpi=80)
    plt.show()
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    buf.close()
    # Save to tensorboard
    with file_writer.as_default():
        tf.summary.image('Denoising Heatmap (PSD cutoff = step/1000000000)', image, step=np.ceil(psd_cutoff*1000000000))

    plt.show()

    # Save figure to Tensorboard
    plt.imshow(tif, vmax=5000)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',  dpi=250)
    plt.title(f'PSD cutoff: {psd_cutoff}')
    plt.show()
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    buf.close()
    # Save to tensorboard
    with file_writer.as_default():
        tf.summary.image('Denoised Image (PSD cutoff = step/1000000000)', image, step=np.ceil(psd_cutoff*1000000000))

    # Save filter_plot to Tensorboard
    with file_writer.as_default():
        if filter_plot is not None:
            tf.summary.image('Denoising Plots (PSD cutoff = step/1000000000)', filter_plot, step=np.ceil(psd_cutoff*1000000000))

    print(f'{idx+1}/{args.num_psd_steps} | PSD cutoff: {psd_cutoff:0.9f}')


# %% ─────────────────────────────────────────────────────────────────────────────
# Merge resulting PDF's
# ────────────────────────────────────────────────────────────────────────────────

# Read all PDF files under logs/ directory
pdf_files = [f for f in os.listdir(logdir) if f.endswith('.pdf')]
if 'denoise_psd_graphs.pdf' in pdf_files:
    raise Exception(f'Danger of Overwriting "denoise_psd_graphs.pdf" already exists under "{logdir}"')
pdf_files = natural_sort(pdf_files)
pdf_files = [os.path.join(logdir, f) for f in pdf_files]

# Categorize PDF files
denoise_psd_graphs = [f for f in pdf_files if 'denoise_psd_graphs' in f]
image_comparison = [f for f in pdf_files if 'image_comparison' in f]

# Merge PDF files
merge_pdfs(denoise_psd_graphs, os.path.join(logdir, 'denoise_psd_graphs.pdf'))
merge_pdfs(image_comparison, os.path.join(logdir, 'image_comparison.pdf'))

# Delete all residual PDF files after merging
for f in pdf_files:
    os.remove(f)
