import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

from helper.collect_traintime_stats import get_scores_and_labels
from calibration_error.bregman_ce import *
from calibration_error.binned_ece import ClasswiseECELoss
from utils.common import create_folder
from utils.constants import *


plt.close()
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("ggplot")
sns.set_palette("tab10")


class ClasswiseECELoss(nn.Module):
    """
    Compute Classwise ECE
    """
    def __init__(self, p=1, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.p = p
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        num_classes = int((torch.max(labels) + 1).item())
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=softmaxes.device)
            labels_in_class = labels.eq(i)  # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += prop_in_bin * (torch.abs(avg_confidence_in_bin - accuracy_in_bin)**self.p)

            if i == 0:
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        return per_class_sce


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
    path_to_results = create_folder(os.path.join(path, 'results'))

    for dataset in ['cifar10']:
        for model_name in ['PreResNet56', 'WideResNet28x10']:
            all_ce_kl = []
            all_ce_l2 = []
            all_ece = []
            for seed in [10, 21, 42, 84]:
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

                ece_classwise = ClasswiseECELoss(p=1, n_bins=15)(scores_test, labels_test)

                # bandwidth = get_bandwidth(scores_test.to(device), device)
                bandwidth = 0.02
                print(f"bandwidth = {bandwidth}")

                # Need to return all_CEf, all_Sf, all_sharpness, instead of means
                ce_kl_classwise = get_classwise_bregman_stats_via_risk_kl(scores_test, labels_test, bandwidth)[0]
                ce_l2_classwise = get_classwise_bregman_stats_via_risk_l2(scores_test, labels_test, bandwidth)[0]

                all_ce_kl.append(ce_kl_classwise)
                all_ce_l2.append(ce_l2_classwise)
                all_ece.append(ece_classwise)

            all_ce_kl_mean = torch.mean(torch.stack(all_ce_kl), dim=0)
            all_ce_l2_mean = torch.mean(torch.stack(all_ce_l2), dim=0)
            all_ece_mean = torch.mean(torch.stack(all_ece), dim=0)

            se_kl = torch.std(torch.stack(all_ce_kl), dim=0) / torch.sqrt(torch.tensor(len(SEEDS)))
            print(se_kl * 100)
            se_l2 = torch.std(torch.stack(all_ce_l2), dim=0) / torch.sqrt(torch.tensor(len(SEEDS)))
            print(se_l2 * 1000)

            ranking_indices_kl = rankdata(all_ce_kl_mean, method='min')
            ranking_indices_l2 = rankdata(all_ce_l2_mean, method='min')
            ranking_indices_ece = rankdata(all_ece_mean, method='min')
            all_kl_string = ""
            all_l2_string = ""
            all_ece_string = ""
            for class_idx in range(scores_test.shape[1]):
                # If you get error, check L107
                all_kl_string += f" & {100 * all_ce_kl_mean[class_idx]:.2f} ({ranking_indices_kl[class_idx]})"
                all_l2_string += f" & {1000 * all_ce_l2_mean[class_idx]:.2f} ({ranking_indices_l2[class_idx]})"
                all_ece_string += f" & {1000 * all_ece_mean[class_idx]:.2f} ({ranking_indices_ece[class_idx]})"

            print(model_name)
            kl_label = r'$\operatorname{CE}_{\mathrm{KL}}$'
            l2_label = r'$\operatorname{CE}_2^2$'
            ece_label = r'$\operatorname{CE}_1$'
            print(f"& {kl_label} \\times 100 {all_kl_string} \\\\")
            print(f"& {l2_label} \\times 1000 {all_l2_string} \\\\")
            print(f"& {ece_label} \\times 1000 {all_ece_string} \\\\")
