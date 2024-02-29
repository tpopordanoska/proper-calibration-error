# This script is used to generate files in the results.zip file uploaded on Zenodo.
import glob
import os

from calibration_error.bregman_ce import *
from utils.common import create_folder, get_scores_and_labels
from utils.constants import *
from collect_traintime_stats import collect_cal_results
import argparse


def init_results():
    results = {}
    for convex_fcn in ['kl', 'l2']:
        for metric in ['ce', 'refinement', 'sharpness']:
            results.update({f'test_{metric}_{convex_fcn}_cw': []})

    for metric in ['loss', 'acc']:
        results.update({f'test_{metric}': []})

    return results


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

    for dataset in DATASETS:
        for model_name in MODEL_NAMES:
            results_per_model = {}
            selected_bandwidths = []
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
                # Find the epoch of best model
                path_to_best_model = glob.glob(os.path.join(path_to_cache_folder, 'epoch_*'))[0]
                best_model_filename = os.path.basename(path_to_best_model)
                epoch = best_model_filename.split("_")[1]
                print(f"Best model saved at epoch: {epoch}")

                # Collect calibration results
                preds_dict = torch.load(os.path.join(path_to_cache_folder, f'preds_{epoch}.pth'))
                scores_test, labels_test = get_scores_and_labels(preds_dict, 'preds_test', 'gt_test')
                bandwidth = get_bandwidth(scores_test.to(device), device)
                selected_bandwidths.append([seed, bandwidth])
                print(f"bandwidth = {bandwidth}, evaluated at {epoch} epoch on test set.")
                collect_cal_results(results, scores_test, labels_test, bandwidth, device, 'test', 'cw')

                # Collect loss and accuracy
                preds_test = torch.argmax(scores_test, dim=1)
                test_acc = preds_test.eq(labels_test).float().mean() * 100.
                logits_test = preds_dict['preds_test']
                test_loss = torch.nn.CrossEntropyLoss()(logits_test, labels_test)
                results['test_acc'].append(test_acc)
                results['test_loss'].append(test_loss)

                torch.save(results, os.path.join(path_to_root_folder, 'best_epoch_test_set_stats.pth'))

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

            final_results['bandwidths'] = selected_bandwidths
            torch.save(results_per_model, os.path.join(path_to_results, f'{dataset}_{model_name}_best_epoch_results_per_model.pth'))
            torch.save(final_results, os.path.join(path_to_results, f'{dataset}_{model_name}_best_epoch_results_avg.pth'))
