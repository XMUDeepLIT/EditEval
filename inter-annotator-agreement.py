import io
from PIL import Image
import imageio
import cv2
import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
import shutil
import os
import matplotlib.pyplot as plt
import krippendorff


def calculate_tau(scores):
    df = pd.DataFrame(scores).T

    kendall_matrix = df.corr(method='kendall')

    tau_values = kendall_matrix.where(np.triu(np.ones(kendall_matrix.shape), k=1).astype(bool)).stack()

    average_kendall_tau = tau_values.mean()
    std_dev_kendall_tau = tau_values.std(ddof=0)

    print(f"Averaged Kendall's τc: {average_kendall_tau:.4f} ± {std_dev_kendall_tau:.4f}")
    return average_kendall_tau, std_dev_kendall_tau


def calculate_spearman(scores):
    df = pd.DataFrame(scores).T

    spearman_matrix = df.corr(method='spearman')

    rho_values = spearman_matrix.where(np.triu(np.ones(spearman_matrix.shape), k=1).astype(bool)).stack()

    average_spearman_rho = rho_values.mean()
    std_dev_spearman_rho = rho_values.std(ddof=0)

    print(f"Averaged Spearman’s ρ: {average_spearman_rho:.4f} ± {std_dev_spearman_rho:.4f}")
    return average_spearman_rho, std_dev_spearman_rho


def calculate_krippendorff_alpha(scores):
    df = pd.DataFrame(scores)

    alpha = krippendorff.alpha(df.values)

    print(f"Krippendorff’s α: {alpha:.4f}")
    return alpha


df = pd.read_csv('labeled_full.csv')

path = 'annotations'
files = sorted(glob.glob(f'{path}/*'))
files = [pd.read_csv(file) for file in files]
files = [pd.merge(df['Index'], file, on='Index', how='left') for file in files]

print('-' * 40 + '\n' + 'Metrics for Textual Faithfulness:')
scores_tf = [list(file['Textual Faithfulness']) for file in files]
_, _ = calculate_tau(scores_tf)
_, _ = calculate_spearman(scores_tf)
_ = calculate_krippendorff_alpha(scores_tf)

print('-' * 40 + '\n' + 'Metrics for Frame Consistency:')
scores_fc = [list(file['Frame Consistency']) for file in files]

calculate_tau(scores_fc)
calculate_spearman(scores_fc)
_ = calculate_krippendorff_alpha(scores_fc)

print('-' * 40 + '\n' + 'Metrics for Video Fidelity:')
scores_fd = [list(file['Video Fidelity']) for file in files]
calculate_tau(scores_fd)
calculate_spearman(scores_fd)
_ = calculate_krippendorff_alpha(scores_fd)