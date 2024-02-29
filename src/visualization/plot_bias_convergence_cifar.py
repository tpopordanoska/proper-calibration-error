import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from visualization.helper.collect_traintime_stats import get_scores_and_labels
from calibration_error.bregman_ce import *
from utils.constants import *
import tikzplotlib

plt.style.use("ggplot")
sns.set_palette("tab10")


def plot_bias_cifar(points, diff_kde_ece_gt_integral, which_convex_fcn, plot_name):
    mean_diff = np.mean(diff_kde_ece_gt_integral, axis=0)
    std_err = np.std(diff_kde_ece_gt_integral, axis=0) / np.sqrt(len(diff_kde_ece_gt_integral))

    plt.plot(points, mean_diff, label=f'{which_convex_fcn}', lw=5)
    plt.fill_between(points, mean_diff - std_err, mean_diff + std_err, alpha=0.25)

    if which_convex_fcn == 'kl':
        label = r"$\operatorname{CE}_{KL}$"
    elif which_convex_fcn == 'l2':
        label = r"$\operatorname{CE}_2^2$"

    y_label = f"{label} bias" if plot_name == 'bias' else f"Relative {label} bias (%)"
    plt.xlabel("Number of points")
    plt.ylabel(y_label)

    ystyle = 'y tick label style={/pgf/number format/.cd, fixed, fixed zerofill, precision=3}' if plot_name == 'bias' \
        else 'y tick label style={/pgf/number format/fixed}'

    extra_params = [
        'scaled y ticks = false',
        f"{ystyle}",
        'scaled x ticks = false',
        'x tick label style={/pgf/number format/fixed}',
        '/pgf/number format/.cd, 1000 sep={}'
    ]

    tikzplotlib.save(
        os.path.join('../figs', f'bias_convergence_{dataset_model_name}_{plot_name}_{which_convex_fcn}.tex'),
        extra_axis_parameters=extra_params
    )

    plt.show()


def plot_ce_values(kde_eces, gt_integral, which_convex_fcn, plot_name):
    ax = plt.subplot(111)
    mean_kde_eces = np.mean(kde_eces, axis=0)
    std_err_kde = np.std(kde_eces, axis=0) / np.sqrt(len(kde_eces))
    gt_mean = np.mean(gt_integral)
    if which_convex_fcn == 'kl':
        label = r"$\operatorname{CE}_{KL}$"
    elif which_convex_fcn == 'l2':
        label = r"$\operatorname{CE}_2^2$"

    ax.plot(points, np.ones(len(mean_kde_eces)) * gt_mean, label="Ground truth", lw=5)
    ax.plot(points, mean_kde_eces, label=label, lw=5)
    ax.fill_between(points, mean_kde_eces - std_err_kde, mean_kde_eces + std_err_kde, alpha=0.25, color=ax.lines[1].get_color())
    plt.xlabel("Number of points")
    plt.ylabel(label)
    plt.legend()

    extra_params = [
        'scaled y ticks = false',
        'y tick label style={/pgf/number format/.cd, fixed, fixed zerofill, precision=3}',
        'scaled x ticks = false',
        'x tick label style={/pgf/number format/fixed}',
        '/pgf/number format/.cd, 1000 sep={}'
    ]

    tikzplotlib.save(
        os.path.join('../figs', f'bias_convergence_{plot_name}_{which_convex_fcn}.tex'),
        extra_axis_parameters=extra_params
    )
    plt.show()


def init_results():
    results = {}
    for convex_fcn in ['kl', 'l2']:
        results.update({f'test_ce_{convex_fcn}_cw': []})

    return results


def collect_results(which_convex_fcn, scores_test, labels_test, diff_est_and_gt, relative_bias, cal_err, gt):
    bandwidth = 0.02
    device = 'mps'
    if which_convex_fcn == 'l2':
        gt_integral = get_classwise_bregman_stats_via_risk_l2(scores_test, labels_test, bandwidth, device)[0].cpu().numpy()
        get_kde_ce_fcn = get_classwise_bregman_stats_via_risk_l2
    elif which_convex_fcn == 'kl':
        gt_integral = get_classwise_bregman_stats_via_risk_kl(scores_test, labels_test, bandwidth, device)[0].cpu().numpy()
        get_kde_ce_fcn = get_classwise_bregman_stats_via_risk_kl
    else:
        raise NotImplementedError

    cal_err_per_seed = []
    points = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000]
    for num_points in points:
        print(num_points)
        scores_sample = scores_test[:num_points]
        targets_sample = labels_test[:num_points]
        ce = get_kde_ce_fcn(scores_sample, targets_sample, bandwidth, device)[0].cpu().numpy()
        cal_err_per_seed.append(ce)

    cal_err_per_seed = np.asarray(cal_err_per_seed)

    cal_err.append(cal_err_per_seed)
    diff_est_and_gt.append(cal_err_per_seed - gt_integral)
    relative_bias.append(((cal_err_per_seed - gt_integral) / gt_integral) * 100)
    gt.append(gt_integral)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models',
                        type=str,
                        default='../../trained_models')
    parser.add_argument('--device',
                        type=str,
                        default='mps')

    args = parser.parse_args()
    args_dict = vars(args)
    path = args_dict['path_to_trained_models']
    device = args_dict['device']
    path_to_results = os.path.join(path, 'results')

    dataset = 'cifar10'
    model_name = 'PreResNet56'
    dataset_model_name = f'{dataset}_{model_name}'
    diff_kde_kl_gt_integral, diff_kde_l2_gt_integral = [], []
    relative_bias_kl, relative_bias_l2 = [], []
    kde_kl, kde_l2 = [], []
    gt_kl, gt_l2 = [], []
    points = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000]
    for seed in SEEDS:
        results = init_results()
        folder_name = f'{dataset}_{model_name}_250_{seed}_output_{seed}'
        path_to_root_folder = os.path.join(path, folder_name)
        print("Root folder path: ", path_to_root_folder)
        cache_path = os.path.join(path_to_root_folder, '*', '*', '*cache*')
        matches = glob.glob(cache_path, recursive=True)
        if len(matches) == 0:
            print('Cache folder not found')
            continue

        path_to_cache_folder = matches[0]
        print("Cache folder path:", path_to_cache_folder)
        # Find the epoch of best model
        path_to_best_model = glob.glob(os.path.join(path_to_cache_folder, 'epoch_*'))[0]
        best_model_filename = os.path.basename(path_to_best_model)
        epoch = best_model_filename.split("_")[1]
        print(f"Best model saved at epoch: {epoch}")

        # Collect calibration results
        preds_dict = torch.load(os.path.join(path_to_cache_folder, f'preds_{epoch}.pth'))
        scores_test, labels_test = get_scores_and_labels(preds_dict, 'preds_test', 'gt_test')
        collect_results('kl', scores_test, labels_test, diff_kde_kl_gt_integral, relative_bias_kl, kde_kl, gt_kl)
        collect_results('l2', scores_test, labels_test, diff_kde_l2_gt_integral, relative_bias_l2, kde_l2, gt_l2)

    # Generates Figure 5
    plot_ce_values(kde_kl, gt_kl, 'kl', dataset_model_name)
    plot_ce_values(kde_l2, gt_l2, 'l2', dataset_model_name)
    plot_bias_cifar(points, diff_kde_kl_gt_integral, 'kl', 'bias')
    plot_bias_cifar(points, diff_kde_l2_gt_integral, 'l2', 'bias')
    plot_bias_cifar(points, relative_bias_kl, 'kl', 'rel_bias')
    plot_bias_cifar(points, relative_bias_l2, 'l2', 'rel_bias')


