# Thesis repository

## Timetable for Meetings

- 26.05.2022

  > 1 Initial Analyisis of FFT using randomly generated data

- 26.05.2022

  > 2 Analyisis of FFT Denoising on the AWAKE Experiment Images

---

## 1. Initial Analyisis of FFT using randomly generated data

### 1. Results

Files are found under `./logs/2022.05.26-FFT_Signal_Denoising_Experiments`

### 1. Running analysis

Set the parameters under `./FFT Analysis/fft_analysis.py`

```python
params = {
  'sample_size': 20,
  'sine_frequency': 3,
  'psd_cutoff': 5,
}
```

Run the python file `./FFT Analysis/fft_analysis.py` using Jupyter Notebook

---

## 2. Analyisis of FFT Denoising on the AWAKE Experiment Images

### 2. Results

Files are found under `./logs/2022.06.02-AWAKE_FFT_Denoising`

### 2. Running analysis

Run the following command in the terminal with desired parameters:

```bash
python fft_denoise_analysis_on_AWAKE.py --beam_window 20 --num_psd_steps 70
```

To get more information about parameters run:

```bash
python fft_denoise_analysis_on_AWAKE.py -h
```
