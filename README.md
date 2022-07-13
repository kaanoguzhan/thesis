# Thesis repository

## Timetable for Meetings of the Task Completions

### 26.05.2022 - [⬇️ Clik here for Task 1][task-1]

> 1 Analyisis of FFT for signal denoising using randomly generated sinusoidal singal data

### 26.05.2022 - [⬇️ Clik here for Task 2][task-2]

> 2 Analyisis of FFT Denoising on the AWAKE Experiment Images

### 23.06.2022 - [⬇️ Clik here for Task 3][task-3]

> 3 Analyisis of the AWAKE experiment images for peak frequency and peak amplitude estimation

### 07.07.2022 - [⬇️ Clik here for Task 4][task-4]

> 4 Covariance Analysis of the AWAKE experiment images

### XX.XX.2022 - [⬇️ Clik here for Task 5][task-5]

> 5 Linearization of time axis for the Streak Camera images

---

## 1. Analyisis of FFT for signal denoising using randomly generated sinusoidal singal data

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

[⬆️ Clik here for Meetings Timetable][meetings-timetable]

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

[⬆️ Clik here for Meetings Timetable][meetings-timetable]

---

## 3. Analyisis of the AWAKE experiment images for peak frequency and peak amplitude estimation

Analysis fpr the following parameters are performed:

- Peak frequency
- Peak amplitude

### 3.1. Pre analysis

1. Load the image
2. Find the marker laser pulse
3. Cut the image after the marker pulse
4. Cut the image around the beam center with the given beam window
5. Split the image into a subimages with the given number window size
6. Perform the Fourier transform (FFT) Denoising on each subimage

### 3.2. Analysis

1. Find the Peak Frequency for each subimage
2. Find the Peak Amplitude at the Peak Frequency for each subimage

### 3.3. Post Analysis

1. Generation and saving of the following plots
   - Histogram of the Peak Frequency
   - Histogram of the Peak Amplitude (at the Peak Frequency)
   - Sub-image index vs. Peak Frequency
   - Sub-image index vs. Peak Amplitude (at the Peak Frequency)
   - Peak Frequency vs. Peak Amplitude

[⬆️ Clik here for Meetings Timetable][meetings-timetable]

---

## 4. Covariance Analysis of the AWAKE experiment images

[⬆️ Clik here for Meetings Timetable][meetings-timetable]

---

## 5 Linearization of time axis for the Streak Camera images

[⬆️ Clik here for Meetings Timetable][meetings-timetable]

---

[//]: # "These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax"
[meetings-timetable]: #timetable-for-meetings-of-the-task-completions
[task-1]: #1-analyisis-of-fft-for-signal-denoising-using-randomly-generated-sinusoidal-singal-data
[task-2]: #2-analyisis-of-fft-denoising-on-the-awake-experiment-images
[task-3]: #3-analyisis-of-the-awake-experiment-images-for-peak-frequency-and-peak-amplitude-estimation
[task-4]: #4-covariance-analysis-of-the-awake-experiment-images
[task-5]: #5-linearization-of-time-axis-for-the-streak-camera-images
