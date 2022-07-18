# %% ─────────────────────────────────────────────────────────────────────────────
#  Imports, Constants, Settings
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
#  Imports
# ────────────────────────────────────────────────────────────
import argparse
import copy
import sys
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from utils.awake_data_loader import AWAKE_DataLoader
from utils.general_utils import in_ipynb

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


print(f'\
Parameters:\n\
----------------------------------------\n\
    Log directory: {logdir}\n\
----------------------------------------\n\
')

# %% ─────────────────────────────────────────────────────────────────────────────
#  Linear Time Axis vs Experiment Timestamps comparison
# ────────────────────────────────────────────────────────────────────────────────

# Load Image using AWAKE_DataLoader
adl = AWAKE_DataLoader('awake_1', [512, 672])
target_image_data = adl.get_data_at_index(3)

streak_img = target_image_data.get_image(linearize_time_axis=False)
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


# %% ─────────────────────────────────────────────────────────────────────────────
#  1-D Interpolation demonstration
# ────────────────────────────────────────────────────────────────────────────────

x_linear = np.array([0, 1, 2, 3, 4, 5])
y_intensity = np.array([50, 3, 4, 25, 10, 30])

x_distorted = np.array([0, 0.8, 1.7, 2.7, 3.8, 5])

f_linear = interpolate.interp1d(x_distorted, y_intensity)

y_interpolated = f_linear(x_linear)

fig = plt.figure(figsize=(13, 8))
plt.plot(x_linear, x_linear, color='tomato', marker='.', markersize=12, label='Linear time')
plt.plot(x_linear, x_distorted, color='mediumseagreen', marker='.', markersize=12, label='Distorted time')
plt.xlabel('Linear time')
plt.ylabel('Reported time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{logdir}/2-1-Interpolation_toy_example_timeline.pdf')


fig = plt.figure(figsize=(13, 8))
plt.plot(x_distorted, y_intensity, color='mediumseagreen', marker='.', markersize=12, label='Intensity on Distorted time')
plt.scatter(x_linear, y_interpolated, color='dodgerblue', s=100, label='Interpolated intensity')
plt.xlabel("Timestamp")
plt.ylabel("Pixel Intensity")
plt.title("1D Interpolation toy example")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{logdir}/2-2-Interpolation_toy_example.pdf')


# %% ─────────────────────────────────────────────────────────────────────────────
#  2-D Interpolation demonstration
# ────────────────────────────────────────────────────────────────────────────────
def streakimdef_old(img, timeaxis):
    timeax_lin = np.linspace(timeaxis[0], timeaxis[-1], len(timeaxis))
    img_t = img.T

    f_interpolate = interpolate.interp1d(timeaxis, img_t)

    imstreak = f_interpolate(timeax_lin)

    return imstreak.T


def streakimdef(img, timeaxis):
    x_linear = np.linspace(timeaxis[0], timeaxis[-1], len(timeaxis))
    img_t = img.T

    # for each row in img, interpolate the intensity values to linear time axis
    f_interpolate = np.zeros_like(img_t)
    for i in range(img_t.shape[0]):
        f_linear = interpolate.interp1d(timeaxis, img_t[i])
        f_interpolate[i] = f_linear(x_linear)

    imstreak_t = f_interpolate.T
    return imstreak_t


# ────────────────────────────────────────────────────────────
# Test - 1
test_img = (np.ones((512, 672)).T * np.linspace(0, 5000, 512)).T

intp_img = copy.deepcopy(test_img)
intp_img = streakimdef_old(intp_img, timeaxis)

# Plot original image
fig = plt.figure(figsize=(13, 13))
plt.imshow(test_img, cmap='gray', vmin=0, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Original Image - Test 1')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-1-1-Original_Image.pdf')

# Plot the difference between the original and the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img-test_img, cmap='coolwarm')
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Difference between original and interpolated image - Test 1')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-1-2-Difference_between_original_and_interpolated_image.pdf')

# Plot the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img, cmap='gray', vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Interpolated Image - Test 1')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-1-3-Interpolated_Image.pdf')
# %%
# ────────────────────────────────────────────────────────────
# Test - 2
test_img = (np.ones((512, 672)) * np.linspace(0, 5000, 672))

intp_img = copy.deepcopy(test_img)
intp_img = streakimdef(intp_img, timeaxis)

# Plot original image
fig = plt.figure(figsize=(13, 13))
plt.imshow(test_img, cmap='gray', vmin=0, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Original Image - Test 2')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-2-1-Original_Image.pdf')

# Plot the difference between the original and the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img-test_img, cmap='coolwarm', vmin=-5e-12, vmax=5e-12)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Difference between original and interpolated image - Test 2')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-2-2-Difference_between_original_and_interpolated_image.pdf')

# Plot the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img, cmap='gray', vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Interpolated Image - Test 2')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-2-3-Interpolated_Image.pdf')

# ────────────────────────────────────────────────────────────
# Test - 3
test_img = np.ones((512, 672))
test_img[10] *= np.linspace(0, 5000, 672)
test_img[11] *= np.linspace(0, 5000, 672)

test_img[50, 0:200] += 5000
test_img[51, 0:200] += 5000

test_img[100, 200:400] += 5000
test_img[101, 200:400] += 5000

test_img[150, 400:672] += 5000
test_img[151, 400:672] += 5000

test_img[200, 0:200] += 5000
test_img[201, 0:200] += 5000

test_img[250, 200:400] += 5000
test_img[251, 200:400] += 5000

test_img[300, 400:672] += 5000
test_img[301, 400:672] += 5000

test_img[325] *= np.linspace(0, 5000, 672)
test_img[326] *= np.linspace(0, 5000, 672)

test_img[350, 0:200] += 5000
test_img[351, 0:200] += 5000

test_img[400, 200:400] += 5000
test_img[401, 200:400] += 5000

test_img[450, 400:672] += 5000
test_img[451, 400:672] += 5000

test_img[500, 0:200] += 5000
test_img[501, 0:200] += 5000

intp_img = copy.deepcopy(test_img)
intp_img = streakimdef(intp_img, timeaxis)

# Plot original image
fig = plt.figure(figsize=(13, 13))
plt.imshow(test_img, cmap='gray', vmin=0, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Original Image - Test 2')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-3-1-Original_Image.pdf')

# Plot the difference between the original and the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img-test_img, cmap='coolwarm')
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Difference between original and interpolated image - Test 3')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-3-2-Difference_between_original_and_interpolated_image.pdf')

# Plot the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img, cmap='gray', vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title('Interpolated Image - Test 3')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-3-3-Interpolated_Image.pdf')

# ────────────────────────────────────────────────────────────
# Test - 4 - AWAKE Image
test_img = adl.get_data_at_index(3).get_image(linearize_time_axis=False)

intp_img = copy.deepcopy(test_img)
intp_img = streakimdef(intp_img, timeaxis)

# Plot original image
fig = plt.figure(figsize=(13, 13))
plt.imshow(test_img, cmap='gray', vmin=0, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title(f'Original Image - {adl.get_data_at_index(3).get_experiment_name()}')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-4-1-Original_Image.pdf')

# Plot the difference between the original and the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img-test_img, cmap='coolwarm', vmin=-5000, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title(f'Difference between original and interpolated image - {adl.get_data_at_index(3).get_experiment_name()}')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-4-2-Difference_between_original_and_interpolated_image.pdf')

# Plot the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img, cmap='gray', vmin=0, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title(f'Interpolated Image - {adl.get_data_at_index(3).get_experiment_name()}')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/3-4-3-Interpolated_Image.pdf')

# ────────────────────────────────────────────────────────────
#  AWAKE Image interpolation using SciPy interpolate.interp2d

test_img = adl.get_data_at_index(3).get_image(linearize_time_axis=False)

intp_img = copy.deepcopy(test_img)
intp_img = streakimdef_old(intp_img, timeaxis)


# Plot original image
fig = plt.figure(figsize=(13, 13))
plt.imshow(test_img, cmap='gray', vmin=0, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title(f'Original Image - {adl.get_data_at_index(3).get_experiment_name()} - Using SciPy interp2d')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/4-1-Original_Image.pdf')

# Plot the difference between the original and the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img-test_img, cmap='coolwarm', vmin=-5000, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title(f'Difference between original and interpolated image - {adl.get_data_at_index(3).get_experiment_name()} - Using SciPy interp2d')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/4-2-Difference_between_original_and_interpolated_image.pdf')

# Plot the interpolated image
fig = plt.figure(figsize=(13, 13))
plt.imshow(intp_img, cmap='gray', vmin=0, vmax=5000)
plt.colorbar(fraction=0.035, pad=0.04)
plt.title(f'Interpolated Image - {adl.get_data_at_index(3).get_experiment_name()} - Using SciPy interp2d')
plt.xlabel('Space')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig(f'{logdir}/4-3-Interpolated_Image.pdf')
