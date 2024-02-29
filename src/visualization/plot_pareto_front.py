import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tikzplotlib
from matplotlib.colors import ListedColormap

from calibration_error.bregman_ce import *

plt.style.use("ggplot")
sns.set_palette("tab10")


# source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
    return is_efficient


def plot(acc_values, ce_values, steps, which_convex_fcn, plot_name):
    acc = acc_values['mean']
    acc_se = acc_values['se']
    ce = ce_values['mean']
    ce_se = ce_values['se']

    costs = torch.stack((100-acc, ce), dim=1)
    flags = is_pareto_efficient_dumb(costs.numpy())

    plt.errorbar(acc, ce, xerr=acc_se, yerr=ce_se,  fmt='none', capsize=2, ecolor='grey', zorder=-1)
    # tikzplotlib has issues with seaborn
    # sns.scatterplot(x=acc, y=ce, hue=flags, s=100, zorder=100, palette="tab10", legend=False)

    color_indices = [0, 1]
    selected_colors = [sns.color_palette('tab10')[i] for i in color_indices]
    cmap = ListedColormap(selected_colors)

    plt.scatter(acc, ce, cmap=cmap, c=flags, s=100, zorder=100, edgecolors='white')
    for i in range(len(acc)):
        if flags[i]:
            plt.text(acc[i] + 0.3, ce[i] - 0.0001, steps[i], ha='left', va='bottom')

    plt.xlabel("Accuracy")
    plt.ylabel(r"$\operatorname{CE}_{KL}$")

    # Invert the y-axis
    ax = plt.gca()
    ax.invert_yaxis()

    extra_params = [
        'scaled y ticks = false',
        'y tick label style={/pgf/number format/.cd, fixed, fixed zerofill, precision=3}'
    ]
    # tikzplotlib.save(
    #     os.path.join('../figs', f'pareto_front_{plot_name}_{which_convex_fcn}.tex'),
    #     extra_axis_parameters=extra_params
    # )
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

    dataset = 'cifar10'
    model_name = 'VGG16BN'
    dataset_model_name = f'{dataset}_{model_name}'
    results = torch.load(os.path.join(path_to_results, f'{dataset_model_name}_results_avg.pth'))
    acc_values = results['val_acc']
    ce_kl_values = results['val_ce_kl_cw']
    ce_l2_values = results['val_ce_l2_cw']
    steps = results['epochs']

    # Generates Figure 7
    plot(acc_values, ce_kl_values, steps, 'kl', dataset_model_name)
    # plot(acc_values, ce_l2_values, steps, 'l2', dataset)
