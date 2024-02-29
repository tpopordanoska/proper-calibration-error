import glob
import os

from calibration_error.bregman_ce import *
from utils.common import create_folder
from netcal.presentation import ReliabilityDiagram
from utils.constants import *
from helper.collect_traintime_stats import collect_cal_results, get_scores_and_labels
import argparse
import tikzplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.close()
mpl.rcParams.update(mpl.rcParamsDefault)
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
            results_per_model = {}
            selected_bandwidths = []
            for seed in ['10']:
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

                for class_idx in range(scores_test.shape[1]):
                    scores_per_class = scores_test[:, class_idx].numpy()
                    labels_per_class = (labels_test == class_idx).numpy().astype(int)

                    diagram = ReliabilityDiagram(10)
                    plt_d = diagram.plot(scores_per_class, labels_per_class, title_suffix=f'Class {class_idx}')

                    tikzplotlib.clean_figure()
                    tikzplotlib_fix_ncols(plt_d)
                    extra_tikzpicture_parameters = ['vertical sep=2cm']
                    tikzplotlib.save(
                        os.path.join('../figs',
                                     f'reliability_diagram_{dataset}_{model_name}_{seed}_class{class_idx}.tex'),
                        extra_groupstyle_parameters=extra_tikzpicture_parameters
                    )
                    plt_d.show()

