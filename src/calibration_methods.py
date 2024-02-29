import argparse
import glob
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F

from calibration_error.bregman_ce import *
from temperature_scaling import ModelWithTemperature
from utils.common import create_folder, get_scores_and_labels
from utils.constants import *

import warnings
warnings.filterwarnings("ignore")

n_bins = 10
bandwidth = 0.02

methods = ['uncal', 'iso', 'temp-nll']
metrics_ce = ['kl', 'l2']
metrics_perf = ['acc', 'nll', 'brier']


def init_results_ce():
    return {method: {metric: [] for metric in metrics_ce} for method in methods}


def init_results_perf():
    return {method: {metric: [] for metric in metrics_perf} for method in methods}


def get_scores_after_isotonic(scores_val, labels_val, scores_test):
    # Fit isotonic regression on the validation data
    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')

    scaler = MinMaxScaler()
    scaled_scores_val = scaler.fit_transform(scores_val)

    calibrated_scores_test = np.zeros_like(scores_test)
    for class_idx in range(scores_test.shape[1]):
        mask = (labels_val == class_idx)
        iso_reg.fit(scaled_scores_val[:, class_idx], mask)
        calibrated_scores_test[:, class_idx] = iso_reg.transform(scores_test[:, class_idx])

    calibrated_scores_test = scaler.inverse_transform(calibrated_scores_test)
    calibrated_scores_test = torch.clamp(torch.tensor(calibrated_scores_test), min=EPS, max=1 - EPS)

    return calibrated_scores_test


def get_scores_after_temp_scale(logits_val, labels_val, logits_test, which_loss='nll'):
    if which_loss == 'nll':
        temp = ModelWithTemperature().set_temperature(logits_val, labels_val)
    elif which_loss == 'brier':
        temp = ModelWithTemperature().set_temperature_brier(logits_val, labels_val)

    calibrated_logits = temp.temperature_scale(logits_test).detach()
    calibrated_scores = F.softmax(calibrated_logits, dim=1)

    calibrated_scores = torch.clamp(calibrated_scores, min=EPS, max=1 - EPS)

    return calibrated_scores


def get_ce(scores_test, labels_test):
    kde_kl = get_classwise_bregman_stats_via_risk_kl(scores_test, labels_test, bandwidth, device)[0].item()
    kde_l2 = get_classwise_bregman_stats_via_risk_l2(scores_test, labels_test, bandwidth, device)[0].item()

    return kde_kl, kde_l2


def get_brier_score(probabilities, labels):
    return (torch.sum((probabilities - labels) ** 2, 1)).mean()


def get_performance_metrics(scores, labels):
    predicted_labels = torch.argmax(scores, dim=1)
    labels_ohe = nn.functional.one_hot(labels, num_classes=scores.shape[1])

    acc = accuracy_score(labels.numpy(), predicted_labels.numpy())
    nll = log_loss(labels, scores)
    brier = get_brier_score(scores, labels_ohe).item()

    return acc, nll, brier


def _format(mean, se):
    return f"${mean}_{{\\pm {se}}}$"


def _format_with_gain(mean, se, gain):
    return f"${mean}_{{\\pm {se}}} (\downarrow {gain} \%)$"


def print_perf_table(results):
    final_results = init_results_perf()
    # Compute mean and standard error across seeds for each key
    for method in results.keys():
        for metric in metrics_perf:
            data = torch.tensor(results[method][metric])
            mean = torch.mean(data, dim=0)
            se = torch.std(data, dim=0) / torch.sqrt(torch.tensor(len(SEEDS)))
            final_results[method][metric] = {
                'mean': mean,
                'se': se
            }

    full_string = f"& {model_name} & "
    for method in methods:
        for metric in metrics_perf:
            mean = f"{100 * final_results[method][metric]['mean'].item():.2f}"
            se = f"{100 * final_results[method][metric]['se'].item():.2f}"
            full_string += _format(mean, se)
            if method == 'temp-nll' and metric == 'brier':
                full_string += " \\\ "
            else:
                full_string += " & "

    print(full_string)


def print_ce_table(results):
    final_results = init_results_ce()
    # Compute mean and standard error across seeds for each key
    for method in results.keys():
        for metric in metrics_ce:
            data = torch.tensor(results[method][metric])
            mean = torch.mean(data, dim=0)
            se = torch.std(data, dim=0) / torch.sqrt(torch.tensor(len(SEEDS)))
            final_results[method][metric] = {
                'mean': mean,
                'se': se
            }

    full_string = f"& {model_name} & "
    for method in methods:
        for metric in metrics_ce:
            mean = f"{100 * final_results[method][metric]['mean'].item():.2f}"
            se = f"{100 * final_results[method][metric]['se'].item():.2f}"
            if method == 'uncal':
                full_string += _format(mean, se)
            else:
                ratio = (final_results['uncal'][metric]['mean'] - final_results[method][metric]['mean']) / \
                        final_results['uncal'][metric]['mean']
                gain = f"{100 * ratio.item():.0f}"
                full_string += _format_with_gain(mean, se, gain)
            if method == 'temp-nll' and metric == 'l2':
                full_string += " \\\ "
            else:
                full_string += " & "

    print(full_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models',
                        type=str,
                        default='../trained_models')
    parser.add_argument('--device',
                        type=str,
                        default='mps')

    args = parser.parse_args()
    args_dict = vars(args)
    path = args_dict['path_to_trained_models']
    device = args_dict['device']
    path_to_results = create_folder(os.path.join(path, 'results'))

    uncalibrated = []
    for dataset in DATASETS:
        for model_name in MODEL_NAMES:
            results_ce = init_results_ce()
            results_perf = init_results_perf()
            for seed in SEEDS:
                folder_name = f'{dataset}_{model_name}_{NUM_EPOCHS}_{seed}_output_{seed}'
                path_to_root_folder = os.path.join(path, folder_name)
                cache_path = os.path.join(path_to_root_folder, '*', '*', '*cache*')
                matches = glob.glob(cache_path, recursive=True)
                if len(matches) == 0:
                    print('Cache folder not found')
                    continue

                path_to_cache_folder = matches[0]

                path_to_preds = glob.glob(os.path.join(path_to_cache_folder, 'preds_*'))[0]
                preds_dict = torch.load(path_to_preds)
                scores_test, labels_test = get_scores_and_labels(preds_dict, 'preds_test', 'gt_test')
                logits_test = preds_dict['preds_test']
                scores_val, labels_val = get_scores_and_labels(preds_dict, 'preds_val', 'gt_val')
                logits_val = preds_dict['preds_val']

                for method in methods:
                    if method == 'uncal':
                        kde_kl, kde_l2 = get_ce(scores_test, labels_test)
                        acc, nll, brier = get_performance_metrics(scores_test, labels_test)
                    elif method == 'temp-nll':
                        scores_cal = get_scores_after_temp_scale(logits_val, labels_val, logits_test)
                        kde_kl, kde_l2 = get_ce(scores_cal, labels_test)
                        acc, nll, brier = get_performance_metrics(scores_cal, labels_test)
                    elif method == 'temp-brier':
                        scores_cal = get_scores_after_temp_scale(logits_val, labels_val, logits_test, 'brier')
                        kde_kl, kde_l2 = get_ce(scores_cal, labels_test)
                        acc, nll, brier = get_performance_metrics(scores_cal, labels_test)
                    elif method == 'iso':
                        scores_cal = get_scores_after_isotonic(scores_val.numpy(), labels_val.numpy(), scores_test.numpy())
                        kde_kl, kde_l2 = get_ce(scores_cal, labels_test)
                        acc, nll, brier = get_performance_metrics(scores_cal, labels_test)

                    results_ce[method]['kl'].append(kde_kl)
                    results_ce[method]['l2'].append(kde_l2)
                    results_perf[method]['acc'].append(acc)
                    results_perf[method]['nll'].append(nll)
                    results_perf[method]['brier'].append(brier)

            print_ce_table(results_ce)
            # print_perf_table(results_perf)
