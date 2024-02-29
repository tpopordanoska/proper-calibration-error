import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib

from calibration_error.bregman_ce import *
from utils.constants import *

plt.style.use("ggplot")
sns.set_palette("tab10")


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def plot_trainval(results, metric, convex_fcn, plot_name):
    is_loss_or_acc = metric == 'loss' or metric == 'acc'
    if is_loss_or_acc:
        train_values = results[f'train_{metric}']['mean']
        val_values = results[f'val_{metric}']['mean']
        train_values_se = results[f'train_{metric}']['se']
        val_values_se = results[f'val_{metric}']['se']
    else:
        train_values = results[f'train_{metric}_{convex_fcn}_cw']['mean']
        val_values = results[f'val_{metric}_{convex_fcn}_cw']['mean']
        train_values_se = results[f'train_{metric}_{convex_fcn}_cw']['se']
        val_values_se = results[f'val_{metric}_{convex_fcn}_cw']['se']
    steps = results['epochs']

    if metric == 'ce':
        metric_label = metric.upper()
    else:
        metric_label = metric.capitalize()
    convex_fcn_label = "" if is_loss_or_acc else f" ({convex_fcn.upper()})"

    fig, ax = plt.subplots()
    error_bar0 = ax.errorbar(steps, train_values, yerr=train_values_se, lw=3)
    error_bar0[0].set_label('Train')
    error_bar1 = ax.errorbar(steps, val_values, yerr=val_values_se, lw=3)
    error_bar1[0].set_label('Validation')

    plt.xlabel('Epochs')
    plt.ylabel(f"{metric_label} {convex_fcn_label}")
    ax.legend()

    tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save(os.path.join('figs', f'{plot_name}_{metric}_{convex_fcn}.tex'))
    plt.show()


def plot_per_split(results, convex_fcn, which_split, plot_name):
    ce_values = results[f'{which_split}_ce_{convex_fcn}_cw']['mean']
    ce_values_se = results[f'{which_split}_ce_{convex_fcn}_cw']['se']
    sharpness_values = results[f'{which_split}_sharpness_{convex_fcn}_cw']['mean']
    sharpness_values_se = results[f'{which_split}_sharpness_{convex_fcn}_cw']['se']
    refinement_values = results[f'{which_split}_refinement_{convex_fcn}_cw']['mean']
    refinement_values_se = results[f'{which_split}_refinement_{convex_fcn}_cw']['se']
    loss_values = results[f'{which_split}_loss']['mean']
    loss_values_se = results[f'{which_split}_loss']['se']
    steps = results['epochs']

    fig, ax = plt.subplots()
    error_bar0 = ax.errorbar(steps, ce_values, yerr=ce_values_se, lw=3)
    error_bar0[0].set_label('CE')
    error_bar1 = ax.errorbar(steps, sharpness_values, yerr=sharpness_values_se, lw=3)
    error_bar1[0].set_label('Sharpness')
    error_bar2 = ax.errorbar(steps, loss_values, yerr=loss_values_se, lw=3)
    error_bar2[0].set_label('Loss')
    error_bar3 = ax.errorbar(steps, refinement_values, yerr=refinement_values_se, lw=3)
    error_bar3[0].set_label('Refinement')

    ax.legend()
    plt.xlabel('Epochs')

    tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save(os.path.join('figs', f'{plot_name}_{which_split}_{convex_fcn}.tex'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models',
                        type=str,
                        default='../../trained_models')

    args = parser.parse_args()
    args_dict = vars(args)
    path = args_dict['path_to_trained_models']
    path_to_results = os.path.join(path, 'results')

    for dataset in DATASETS:
        for model_name in MODEL_NAMES:
            dataset_model_name = f'{dataset}_{model_name}'
            results = torch.load(os.path.join(path_to_results, f'{dataset_model_name}_results_avg.pth'))
            plot_trainval(results, 'ce', 'kl', dataset_model_name)
            plot_trainval(results, 'ce', 'l2', dataset_model_name)
            plot_trainval(results, 'refinement', 'kl', dataset_model_name)
            plot_trainval(results, 'refinement', 'l2', dataset_model_name)
            plot_trainval(results, 'sharpness', 'kl', dataset_model_name)
            plot_trainval(results, 'sharpness', 'l2', dataset_model_name)
            plot_trainval(results, 'loss', '', dataset_model_name)
            plot_trainval(results, 'acc', '', dataset_model_name)

            plot_per_split(results, 'kl', 'train', dataset_model_name)
            plot_per_split(results, 'l2', 'train', dataset_model_name)
            plot_per_split(results, 'kl', 'val', dataset_model_name)
            plot_per_split(results, 'l2', 'val', dataset_model_name)
