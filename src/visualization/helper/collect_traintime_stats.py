# This script is used to generate files in the results.zip file uploaded on Zenodo.
import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

from calibration_error.bregman_ce import *
from utils.common import create_folder, get_scores_and_labels
from utils.constants import *


def collect_cal_results(results, scores, labels, bandwidth, device, split, type):
    if split == 'train':
        # Subsample 10k scores and labels
        idx = torch.randperm(len(scores))[:10000]
        scores = scores[idx, :]
        labels = labels[idx]
    if type == 'cw':
        get_results_fcn_kl = get_classwise_bregman_stats_via_risk_kl
        get_results_fcn_l2 = get_classwise_bregman_stats_via_risk_l2
    elif type == 'cncl':
        get_results_fcn_kl = get_bregman_stats_via_risk_kl
        get_results_fcn_l2 = get_bregman_stats_via_risk_l2
    else:
        raise NotImplementedError

    ce_kl, refinement_kl, sharpness_kl = get_results_fcn_kl(scores, labels, bandwidth, device)
    ce_l2, refinement_l2, sharpness_l2 = get_results_fcn_l2(scores, labels, bandwidth, device)

    results[f'{split}_ce_kl_{type}'].append(ce_kl)
    results[f'{split}_refinement_kl_{type}'].append(refinement_kl)
    results[f'{split}_sharpness_kl_{type}'].append(sharpness_kl)
    results[f'{split}_ce_l2_{type}'].append(ce_l2)
    results[f'{split}_refinement_l2_{type}'].append(refinement_l2)
    results[f'{split}_sharpness_l2_{type}'].append(sharpness_l2)


def init_results():
    results = {}
    for split in ['train', 'val', 'test']:
        for convex_fcn in ['kl', 'l2']:
            for metric in ['ce', 'refinement', 'sharpness']:
                results.update({f'{split}_{metric}_{convex_fcn}_cw': []})

    for split in ['train', 'val']:
        for metric in ['loss', 'acc']:
            results.update({f'{split}_{metric}': []})

    return results


def get_values(path_to_tfevents_file, metric):
    event_loss_train = EventAccumulator(path_to_tfevents_file)
    event_loss_train.Reload()
    values = [tv.value for tv in event_loss_train.Scalars(metric)]

    return values


def get_values_from_tfevents(path_to_cache_folder):
    path_to_tfevents_folder = os.path.join(path_to_cache_folder, 'tb_logs', 'run')

    path_to_tfevents_loss_train_file = glob.glob(os.path.join(path_to_tfevents_folder, 'Loss_train', 'events*'))[0]
    path_to_tfevents_loss_val_file = glob.glob(os.path.join(path_to_tfevents_folder, 'Loss_val', 'events*'))[0]
    path_to_tfevents_acc_val_file = glob.glob(os.path.join(path_to_tfevents_folder, 'events*'))[0]

    train_loss = get_values(path_to_tfevents_loss_train_file, 'Loss')
    val_loss = get_values(path_to_tfevents_loss_val_file, 'Loss')
    acc_val = get_values(path_to_tfevents_acc_val_file, 'Metrics/accuracy')

    return train_loss, val_loss, acc_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models',
                        type=str,
                        default='../../../trained_models')
    parser.add_argument('--device',
                        type=str,
                        default='mps')
    parser.add_argument('--eval_every',
                        type=int,
                        default=10)

    args = parser.parse_args()
    args_dict = vars(args)
    path = args_dict['path_to_trained_models']
    path_to_results = create_folder(os.path.join(path, 'results'))
    device = args_dict['device']
    eval_every = args_dict['eval_every']

    for dataset in DATASETS:
        for model_name in MODEL_NAMES:
            results_per_model = {}
            is_bandwidth_evaluated = False
            for seed in SEEDS:
                results = init_results()
                folder_name = f'{dataset}_{model_name}_{NUM_EPOCHS}_{seed}_output_{seed}'
                path_to_root_folder = os.path.join(path, folder_name)
                print("Root folder path: ", path_to_root_folder)
                cache_path = os.path.join(path_to_root_folder, '*', '*', '*cache*')
                matches = glob.glob(cache_path, recursive=True)
                if len(matches) == 0:
                    print('Cache folder not found')
                    continue

                path_to_cache_folder = matches[0]
                print("Cache folder path:", path_to_cache_folder)
                loss_train, loss_val, acc_val = get_values_from_tfevents(path_to_cache_folder)

                for epoch in tqdm(range(eval_every, NUM_EPOCHS, eval_every)):
                    if not is_bandwidth_evaluated:
                        # Select bandwidth from val set once per model from reasonably trained model
                        preds_at_half_train = torch.load(os.path.join(path_to_cache_folder, f'preds_125.pth'))
                        scores_val_at_half_train, _ = get_scores_and_labels(preds_at_half_train, 'preds_val', 'gt_val')
                        bandwidth = get_bandwidth(scores_val_at_half_train.to(device), device)
                        print(f"bandwidth = {bandwidth}, evaluated at 125th epoch on val set.")
                        is_bandwidth_evaluated = True

                    preds_dict = torch.load(os.path.join(path_to_cache_folder, f'preds_{epoch}.pth'))
                    scores_val, labels_val = get_scores_and_labels(preds_dict, 'preds_val', 'gt_val')
                    scores_train, labels_train = get_scores_and_labels(preds_dict, 'preds_train', 'gt_train')
                    scores_test, labels_test = get_scores_and_labels(preds_dict, 'preds_test', 'gt_test')

                    collect_cal_results(results, scores_train, labels_train, bandwidth, device, 'train', 'cw')
                    collect_cal_results(results, scores_val, labels_val, bandwidth, device, 'val', 'cw')
                    collect_cal_results(results, scores_test, labels_test, bandwidth, device, 'test', 'cw')

                    preds_train = torch.argmax(scores_train, dim=1)
                    train_acc = preds_train.eq(labels_train).float().mean() * 100.
                    results[f'train_acc'].append(train_acc)
                    results[f'train_loss'].append(loss_train[epoch])
                    results[f'val_loss'].append(loss_val[epoch])
                    results[f'val_acc'].append(acc_val[epoch])

                torch.save(results, os.path.join(path_to_root_folder, 'results.pth'))
                # Combine results dict into results_per_model dict
                for key in results.keys():
                    if key not in results_per_model:
                        results_per_model[key] = []
                    results_per_model[key].append(results[key])

            final_results = {}
            # Compute mean and standard error across seeds for each key
            for key in results_per_model.keys():
                data = torch.tensor(results_per_model[key])
                mean = torch.mean(data, dim=0)
                se = torch.std(data, dim=0) / torch.sqrt(torch.tensor(len(SEEDS)))
                print(f'{key}: mean={mean}, se={se}')
                final_results[key] = {
                    'mean': mean,
                    'se': se
                }

            final_results['bandwidth'] = bandwidth
            final_results['epochs'] = list(range(eval_every, NUM_EPOCHS, eval_every))
            torch.save(results_per_model, os.path.join(path_to_results, f'{dataset}_{model_name}_results_per_model.pth'))
            torch.save(final_results, os.path.join(path_to_results, f'{dataset}_{model_name}_results_avg.pth'))
