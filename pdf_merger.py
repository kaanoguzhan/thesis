# %%
# This script is used to merge PDF files with same name across different experiments.

import os
from utils.general_utils import natural_sort, merge_pdfs

EXPERIMENTS_LOG_PATH = os.path.join('logs', 'AWAKE_PSD_Peak_Analysis')

pdf_names = [
    "6_clean_psd_histogram_low_res.pdf",
    "8_phase_at_peaks.pdf",
    "9_freq_at_peaks_vs_beam_window.pdf",
    "9_phase_at_peaks_vs_beam_window.pdf",
    "9_freq_at_peaks_vs_phase_at_peaks.pdf",
]

# ────────────────────────────────────────────────────────────────────────────────


def merge_pdfs_across_experiments(experiment_log_path: str, pdf_name: str):
    # list all the files in the experiment log folder
    files = os.listdir(experiment_log_path)

    # Read all PDF files under EXPERIMENT_LOG_PATH directory
    experiment_folders = [f for f in files if os.path.isdir(os.path.join(experiment_log_path, f))]
    experiment_folders = natural_sort(experiment_folders, reverse=True)

    pdf_files = []
    for experiment_folder in experiment_folders:
        pdf_file = os.path.join(experiment_log_path, experiment_folder, pdf_name)
        pdf_files.append(pdf_file)

    print(f'Merging {len(pdf_files)} PDF files...')

    merge_pdfs(pdf_paths=pdf_files, output=os.path.join(experiment_log_path, f'Merge_{pdf_name}'))


for pdf_name in pdf_names:
    print(f'Combining "{pdf_name}" across experiments...')
    merge_pdfs_across_experiments(experiment_log_path=EXPERIMENTS_LOG_PATH, pdf_name=pdf_name)
    print(f'Done!')
