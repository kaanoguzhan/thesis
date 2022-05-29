# %%
# Imports and settings
import numpy as np
import matplotlib.pyplot as plt
import os

params = {
    'sample_size': 20,
    'sine_frequency': 3,
    'psd_cutoff': 5,
}

plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 20
plt.rcParams['lines.markersize'] = 7
plt.rcParams['legend.loc'] = 'upper right'

experiment_name = f'{params["sample_size"]}_samples_at_{params["sine_frequency"]}Hz'
# Make dir with experiment name
os.makedirs(f'experiments/{experiment_name}', exist_ok=True)

# %%
# Signal generation
print(f'Generating signal with frequency 120 Hz and using {params["sample_size"]} samples')

# Underlying Original signal
t_original = np.arange(0, 1, 1/1000)
f_original = np.sin(params['sine_frequency'] * 2 * np.pi * t_original)  # np.sin(2 * np.pi * 120 * t)  # Sum of 2 frequencies

# Sample Signal
dt = 1/params['sample_size']
t_experiment = np.arange(0, 1, dt)
f_sample = np.sin(params['sine_frequency'] * 2 * np.pi * t_experiment)
f_clean = f_sample.copy()
f_sample = f_sample + np.random.randn(len(f_sample))  # Add some noise

fig, axs = plt.subplots(4, 1, figsize=(20, 22))
fig.tight_layout()
plt.subplots_adjust(hspace=0.3)
# set xlim & ylim for all plots
for ax in axs.flatten():
    ax.set_ylim(-3, 3)
    ax.set_xlim(t_experiment[0], t_experiment[-1])

plt.sca(axs[0])
plt.plot(t_original, f_original, color='g', label='Original signal')
plt.grid()
plt.title('Original signal')
plt.legend()

plt.sca(axs[1])
plt.plot(t_original, f_original, color='g', label='Original signal')
plt.plot(t_experiment, f_clean, 'o', color='r', label='Clean signal')
plt.grid()
plt.title('Clean signal')
plt.legend()

plt.sca(axs[2])
plt.plot(t_experiment, f_sample, 'o', color='k', label='Noisy signal')
plt.plot(t_experiment, f_clean, 'o', color='r', label='Clean signal')
plt.grid()
plt.title('Noisy signal vs Clean signal')
plt.legend()

plt.sca(axs[3])
plt.plot(t_experiment, f_sample, color='k')
plt.plot(t_experiment, f_sample, 'o', color='k', label='Noisy signal')
plt.grid()
plt.title('Noisy signal')
plt.legend()

plt.legend()
plt.savefig(f'experiments/{experiment_name}/1_noisy_signal_generation.pdf')

# %%
# Compute Fourier Coefficients of the f_sample signal
# Fourier Coefficients have magnitude and phase
Fn = np.fft.fft(f_sample, params['sample_size'])

# %%
# Use PSD to filter out noise

# Compute Power Spectral Denisty
p_s_d = Fn * np.conj(Fn) / params['sample_size']

# Find all frequencies with large power
indices = p_s_d > params['psd_cutoff']

# Zero out smaller Fourier coefficients and their corresponding frequencies
Fn = Fn * indices
PSDClean = p_s_d * indices

# Apply Inverse FFT
ffilt = np.fft.ifft(Fn)

freq = np.arange(params['sample_size'])

# %%
# Plotting denoising results
fig, axs = plt.subplots(4, 1, figsize=(20, 22))
fig.tight_layout()
plt.subplots_adjust(hspace=0.3)

plt.sca(axs[0])
plt.plot(t_experiment, f_sample, color='k')
plt.plot(t_experiment, f_sample, 'o', color='k', label='Noisy signal')
plt.xlim(t_experiment[0], t_experiment[-1])
plt.ylim(-3, 3)
plt.title('Noisy signal')
plt.grid()
plt.legend()

plt.sca(axs[1])
plt.plot(freq, p_s_d[freq], 'o', color='b', linewidth=2, label='Noisy signal')
plt.plot(freq, p_s_d[freq], color='b')
plt.plot([freq[0], freq[-1]], [params['psd_cutoff'], params['psd_cutoff']], '--', color='tab:orange', label='PSD cutoff')
plt.xlim(freq[0], freq[-1])
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectrum |Fn|^2')
plt.legend()

plt.sca(axs[2])
plt.plot(freq, PSDClean[freq], color='darkred')
plt.plot(freq, PSDClean[freq], 'o', color='darkred', linewidth=2, label='Filtered signal')
plt.plot([freq[0], freq[-1]], [params['psd_cutoff'], params['psd_cutoff']], '--', color='tab:orange', label='PSD cutoff')
plt.xlim(freq[0], freq[-1])
plt.title('Power Spectral Density after filtering')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectrum |Fn|^2')
plt.legend()

plt.sca(axs[3])
plt.plot(t_experiment, ffilt, 'o', color='r', linewidth=2, label='Filtered signal')
plt.plot(t_original, f_original, color='g', label='Original signal')
plt.xlim(t_experiment[0], t_experiment[-1])
plt.ylim(-3, 3)
plt.title('Filtered signal vs Original signal')
plt.grid()
plt.legend()

plt.savefig(f'experiments/{experiment_name}/2_denoising_results.pdf')
