import copy
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import tikzplotlib
from tqdm import tqdm

from calibration_error.binned_ece import *
from calibration_error.bregman_ce import *
from utils.constants import EPS

plt.close()
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("ggplot")
sns.set_palette("tab10")


def get_bandwidth_synthetic(num_points):
    if num_points <= 1000:
        return 0.01
    elif num_points <= 5000:
        return 0.001
    elif num_points <= 10000:
        return 0.0005
    else:
        return 0.0001


def sample_points(num_samples, num_classes, temp1, temp2):
    logits_temp1 = []
    samples = []
    labels = []
    for _ in range(num_samples):
        # Sample a point from the simplex
        sample = sample_from_simplex(num_classes, 1)
        # Temp scale it to simulate our setting
        logit = temp_scale(torch.tensor(sample), temp1).unsqueeze(0)
        scores = torch.softmax(logit, dim=1)[0]
        logits_temp1.append(logit.numpy())
        samples.append(scores.numpy())
        # Sample y according to that point
        labels.append(np.random.choice(np.arange(0, num_classes), p=scores.numpy()))

    logits_temp1 = torch.tensor(np.array(logits_temp1)).squeeze()
    pred_scores_temp1 = torch.tensor(np.array(samples)).type(torch.DoubleTensor)
    targets = torch.tensor(np.array(labels).astype('int64'))

    # Second temp scaling
    logits_temp2 = temp_scale(pred_scores_temp1, temp2)
    pred_scores_temp2 = torch.softmax(logits_temp2, dim=1)

    # Scores cannot be exactly 0 or 1
    pred_scores_temp1 = torch.clamp(pred_scores_temp1, min=EPS, max=1 - EPS)
    pred_scores_temp2 = torch.clamp(pred_scores_temp2, min=EPS, max=1 - EPS)

    return pred_scores_temp1, pred_scores_temp2, logits_temp1, logits_temp2, targets


# from: https://github.com/HLT-ISTI/QuaPy/blob/ee6af04abdd261d688f1f38a6c20193d44b04198/quapy/functional.py#L98
def sample_from_simplex(n_classes, size=1):
    """
    Implements the `Kraemer algorithm <http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf>`_
    for sampling uniformly at random from the unit simplex. This implementation is adapted from this
    `post <https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex>_`.

    Args:
        n_classes: integer, number of classes (dimensionality of the simplex)
        size: number of samples to return

    Returns: `np.ndarray` of shape `(size, n_classes,)` if `size>1`, or of shape `(n_classes,)` otherwise

    """
    if n_classes == 2:
        u = np.random.rand(size)
        u = np.vstack([1-u, u]).T
    else:
        u = np.random.rand(size, n_classes-1)
        u.sort(axis=-1)
        _0s = np.zeros(shape=(size, 1))
        _1s = np.ones(shape=(size, 1))
        a = np.hstack([_0s, u])
        b = np.hstack([u, _1s])
        u = b-a
    if size == 1:
        u = u.flatten()
    return u


def temp_scale(scores, temperature):
    logits = inv_softmax(scores)
    return logits / temperature


def inv_softmax(x, c=torch.log(torch.tensor(10))):
    return torch.log(x) + c


def plot_ce(which_convex_fcn, type='canonical'):
    num_classes, num_samples, num_bins, temp1, temp2, bandwidth = get_setup()
    if type == 'classwise':
        get_bregman_ce_fcn = get_classwise_bregman_ce
    else:
        get_bregman_ce_fcn = get_bregman_ce

    pred_scores_temp1, pred_scores_temp2, _, _, _ = sample_points(100000, num_classes, temp1, temp2)
    if which_convex_fcn == 'l2':
        convex_fcn = l2_norm
        gt_integral = get_l2_gt_integral(pred_scores_temp1, pred_scores_temp2, type)
    elif which_convex_fcn == 'kl':
        convex_fcn = negative_entropy
        gt_integral = get_kl_gt_integral(pred_scores_temp1, pred_scores_temp2, type)
    else:
        raise NotImplementedError

    # Sample new points
    pred_scores_temp1, pred_scores_temp2, _, logits_temp2, targets = sample_points(num_samples, num_classes, temp1, temp2)

    ax = plt.subplot(111)
    points = [50, 100, 500, 1000, 5000, 10000]

    kde_eces = []
    for num_points in points:
        print(num_points)
        scores_sample = pred_scores_temp2[:num_points]
        targets_sample = targets[:num_points]
        # For automatic bandwidth selection: bandwidth = get_bandwidth(scores_sample, 'cpu')
        kde = get_bregman_ce_fcn(convex_fcn, scores_sample, targets_sample, bandwidth)
        kde_eces.append(kde)

    ax.plot(points, np.ones(len(kde_eces)) * gt_integral.numpy(), label="Ground truth")
    ax.plot(points, torch.stack(kde_eces).detach().numpy(), label=f"KDE-ECE")

    plt.legend()
    plt.title(f"{which_convex_fcn.upper()}. Num classes: {num_classes}")
    plt.show()


def get_l2_gt_integral(pred_scores_temp1, pred_scores_temp2, type):
    if type == 'classwise':
        classwise_gt = []
        for i in range(pred_scores_temp2.shape[1]):
            pred_scores_temp2_i = pred_scores_temp2[:, i].unsqueeze(-1)
            pred_scores_temp2_i = torch.cat((1 - pred_scores_temp2_i, pred_scores_temp2_i), dim=1)
            pred_scores_temp1_i = pred_scores_temp1[:, i].unsqueeze(-1)
            pred_scores_temp1_i = torch.cat((1 - pred_scores_temp1_i, pred_scores_temp1_i), dim=1)
            classwise_gt.append(torch.sum(abs(pred_scores_temp1_i - pred_scores_temp2_i)**2) / len(pred_scores_temp1_i))
        classwise_gt = torch.stack(classwise_gt)
        gt_integral = torch.mean(classwise_gt)
    else:
        gt_integral = torch.sum(abs(pred_scores_temp1 - pred_scores_temp2)**2) / len(pred_scores_temp1)
    print("Ground truth: ", gt_integral)

    return gt_integral


def get_kl_gt_integral(pred_scores_temp1, pred_scores_temp2, type):
    p = pred_scores_temp1
    q = pred_scores_temp2
    if type == 'classwise':
        classwise_gt = []
        for i in range(pred_scores_temp2.shape[1]):
            q_i = q[:, i].unsqueeze(-1)
            q_i = torch.cat((1 - q_i, q_i), dim=1)
            p_i = p[:, i].unsqueeze(-1)
            p_i = torch.cat((1 - p_i, p_i), dim=1)
            N = len(p_i)
            inner_sum = 0
            for h in range(N):
                inner_sum += torch.dot(p_i[h], torch.log(p_i[h] / q_i[h]))
            classwise_gt.append(inner_sum / N)
        classwise_gt = torch.stack(classwise_gt)
        gt_integral = torch.mean(classwise_gt)
    else:
        N = len(p)
        final_sum = 0
        for h in range(N):
            final_sum += torch.dot(p[h], torch.log(p[h]/q[h]))
        gt_integral = final_sum / N
    print("Ground truth: ", gt_integral)

    return gt_integral


def test_l2_vs_kl():
    kde_kl = []
    kde_l2 = []
    classes = 10
    bandwidth = 0.005
    for _ in tqdm(range(500)):
        pred_scores_temp1, pred_scores_temp2, _, _, targets = sample_points(1000, classes, 0.9, 0.3)
        kde_kl.append(get_classwise_bregman_stats_via_risk_kl(pred_scores_temp2, targets, bandwidth)[0].detach().numpy())
        kde_l2.append(get_classwise_bregman_stats_via_risk_l2(pred_scores_temp2, targets, bandwidth)[0].detach().numpy())

    plt.scatter(kde_kl, kde_l2)
    plt.xlabel("CE (KL)")
    plt.ylabel("CE (L2)")
    plt.xlim(0.5, 0.3)
    plt.ylim(0.5, 0.3)

    extra_params = [
        'scaled y ticks = false',
        'y tick label style={/pgf/number format/fixed}',
        'scaled x ticks = false',
        'x tick label style={/pgf/number format/fixed}',
        '/pgf/number format/.cd, 1000 sep={}'
    ]

    tikzplotlib.save(
        os.path.join('figs', f'l2_vs_kl_{classes}_classes.tex'),
        extra_axis_parameters=extra_params,
        )
    plt.show()


def compare_with_binning(which_convex_fcn, num_repeats=2):
    num_classes, _, num_bins, temp1, temp2, _ = get_setup()
    pred_scores_temp1, pred_scores_temp2, _, _, targets = sample_points(50000, num_classes, temp1, temp2)
    if which_convex_fcn == 'l2':
        get_bregman_ce_fcn = get_classwise_bregman_stats_via_risk_l2
        bin_fcn = ClasswiseECELoss(p=2, n_bins=num_bins)
        gt_integral = get_l2_gt_integral(pred_scores_temp1, pred_scores_temp2, 'classwise').numpy()
    elif which_convex_fcn == 'kl':
        get_bregman_ce_fcn = get_classwise_bregman_stats_via_risk_kl
        bin_fcn = ECELoss_KL(num_bins)
        gt_integral = get_kl_gt_integral(pred_scores_temp1, pred_scores_temp2, 'classwise').numpy()
    else:
        raise NotImplementedError

    points = [100, 500, 1000, 3000, 5000, 8000, 10000]
    all_kdes = []
    all_bins = []
    for _ in tqdm(range(num_repeats)):
        kde_list, bin_list, = [], []
        pred_scores_temp1, pred_scores_temp2, logits_temp1, logits_temp2, targets = sample_points(15000, num_classes, temp1, temp2)

        for num_points in points:
            print(num_points)
            scores_sample = pred_scores_temp2[:num_points]
            targets_sample = targets[:num_points]
            logits_temp2_sample = logits_temp2[:num_points]
            bandwidth = get_bandwidth_synthetic(num_points)
            kde = get_bregman_ce_fcn(scores_sample, targets_sample, bandwidth, 'mps')[0].item()
            kde_list.append(kde)
            bin = bin_fcn(logits_temp2_sample, targets_sample).item()
            bin_list.append(bin)

        all_kdes.append(np.asarray(kde_list))
        all_bins.append(np.asarray(bin_list))

    plot_bin_vs_kde(all_kdes, all_bins, gt_integral, points, which_convex_fcn.upper())


def plot_bin_vs_kde(kde_list, bin_list, gt, points, which_convex_fcn):
    mean_kde = np.mean(kde_list, axis=0)
    std_err_kde = np.std(kde_list, axis=0) / np.sqrt(len(kde_list))

    mean_bin = np.mean(bin_list, axis=0)
    std_error_bin = np.std(bin_list, axis=0) / np.sqrt(len(bin_list))

    plt.errorbar(points, mean_kde, yerr=std_err_kde, fmt='o-', label=f'KDE-{which_convex_fcn}')
    plt.fill_between(points, mean_kde - std_err_kde, mean_kde + std_err_kde, alpha=0.25)

    plt.errorbar(points, mean_bin, yerr=std_error_bin, fmt='o-', label=f'BIN-{which_convex_fcn}')
    plt.fill_between(points, mean_bin - std_error_bin, mean_bin + std_error_bin, alpha=0.25)

    plt.plot(points, np.ones(len(mean_kde)) * gt, label=f'GT')

    plt.legend()
    plt.xlabel("Number of points")
    plt.ylabel("Calibration error")

    extra_params = [
        'scaled y ticks = false',
        'y tick label style={/pgf/number format/fixed}',
        'scaled x ticks = false',
        'x tick label style={/pgf/number format/fixed}',
        'xtick = {0, 5000, 10000, 15000}',
        '/pgf/number format/.cd, 1000 sep={}'
    ]

    tikzplotlib.clean_figure()
    tikzplotlib.save(
        os.path.join('figs', f'bin_vs_kde_{which_convex_fcn.lower()}.tex'),
        extra_axis_parameters=extra_params,
        )
    plt.show()


def compare_two_impl(which_convex_fcn, num_repeats):
    num_classes, _, num_bins, temp1, temp2, bandwidth = get_setup()
    pred_scores_temp1, pred_scores_temp2, _, _, targets = sample_points(50000, num_classes, temp1, temp2)
    if which_convex_fcn == 'l2':
        convex_fcn = l2_norm
        get_bregman_ce_via_risk_fcn = get_classwise_bregman_stats_via_risk_l2
        gt_integral = get_l2_gt_integral(pred_scores_temp1, pred_scores_temp2, 'classwise').numpy()
    elif which_convex_fcn == 'kl':
        convex_fcn = negative_entropy
        get_bregman_ce_via_risk_fcn = get_classwise_bregman_stats_via_risk_kl
        gt_integral = get_kl_gt_integral(pred_scores_temp1, pred_scores_temp2, 'classwise').numpy()
    else:
        raise NotImplementedError

    points = [100, 500, 1000, 3000, 5000, 8000, 10000]
    all_kdes_via_risk = []
    all_kdes_direct = []
    for _ in tqdm(range(num_repeats)):
        kde_via_risk_list, kde_direct_list, = [], []
        pred_scores_temp1, pred_scores_temp2, logits_temp1, logits_temp2, targets = sample_points(15000, num_classes, temp1, temp2)

        for num_points in points:
            print(num_points)
            scores_sample = pred_scores_temp2[:num_points]
            targets_sample = targets[:num_points]
            bandwidth = get_bandwidth_synthetic(num_points)
            kde_via_risk = get_bregman_ce_via_risk_fcn(scores_sample, targets_sample, bandwidth, 'mps')[0].item()
            kde_via_risk_list.append(kde_via_risk)
            kde_direct = get_classwise_bregman_ce(convex_fcn, scores_sample, targets_sample, bandwidth, 'mps').item()
            kde_direct_list.append(kde_direct)

        all_kdes_via_risk.append(np.asarray(kde_via_risk_list))
        all_kdes_direct.append(np.asarray(kde_direct_list))

    plot_two_impl(all_kdes_via_risk, all_kdes_direct, gt_integral, points, which_convex_fcn.upper())


def plot_two_impl(kde_via_risk_list, kde_direct_list, gt, points, which_convex_fcn):
    mean_kde = np.mean(kde_via_risk_list, axis=0)
    std_err_kde = np.std(kde_via_risk_list, axis=0) / np.sqrt(len(kde_via_risk_list))

    mean_bin = np.mean(kde_direct_list, axis=0)
    std_error_bin = np.std(kde_direct_list, axis=0) / np.sqrt(len(kde_direct_list))

    plt.errorbar(points, mean_kde, yerr=std_err_kde, fmt='o-', label=f'KDE-{which_convex_fcn} via risk ')
    plt.fill_between(points, mean_kde - std_err_kde, mean_kde + std_err_kde, alpha=0.25)

    plt.errorbar(points, mean_bin, yerr=std_error_bin, fmt='o-', label=f'KDE-{which_convex_fcn} direct')
    plt.fill_between(points, mean_bin - std_error_bin, mean_bin + std_error_bin, alpha=0.25)

    plt.plot(points, np.ones(len(mean_kde)) * gt, label=f'GT')
    plt.legend()
    plt.xlabel("Number of points")
    plt.ylabel("Calibration error")

    extra_params = [
        'scaled y ticks = false',
        'y tick label style={/pgf/number format/fixed}',
        'scaled x ticks = false',
        'x tick label style={/pgf/number format/fixed}',
        'xtick = {0, 5000, 10000, 15000}',
        '/pgf/number format/.cd, 1000 sep={}'
    ]

    tikzplotlib.clean_figure()
    tikzplotlib.save(
        os.path.join('figs', f'two_impl_{which_convex_fcn.lower()}.tex'),
        extra_axis_parameters=extra_params,
        )

    plt.show()


def collect_results_for_nclasses(which_convex_fcn, num_classes, temp1, temp2, num_samples, num_repeats):
    pred_scores_temp1, pred_scores_temp2, _, _, _ = sample_points(50000, num_classes, temp1, temp2)
    if which_convex_fcn == 'l2':
        get_bregman_ce_fcn = get_classwise_bregman_stats_via_risk_l2
        gt_integral = get_l2_gt_integral(pred_scores_temp1, pred_scores_temp2, 'classwise').numpy()
    elif which_convex_fcn == 'kl':
        get_bregman_ce_fcn = get_classwise_bregman_stats_via_risk_kl
        gt_integral = get_kl_gt_integral(pred_scores_temp1, pred_scores_temp2, 'classwise').numpy()
    else:
        raise NotImplementedError

    diff_kde_ece_gt_integral = []
    relative_bias = []
    for _ in tqdm(range(num_repeats)):
        # Sample new points
        pred_scores_temp1, pred_scores_temp2, _, logits_temp2, targets = sample_points(num_samples, num_classes, temp1, temp2)
        # points = [100, 500, 1000, 5000]
        points = [100, 500, 1000, 3000, 5000, 8000, 10000]
        kde_eces = []
        for num_points in points:
            print(num_points)
            scores_sample = pred_scores_temp2[:num_points]
            targets_sample = targets[:num_points]
            bandwidth = get_bandwidth_synthetic(num_points)
            kde = get_bregman_ce_fcn(scores_sample, targets_sample, bandwidth, 'mps')[0].item()
            kde_eces.append(kde)

        kde_eces = np.asarray(kde_eces)
        diff_kde_ece_gt_integral.append(kde_eces - gt_integral)
        relative_bias.append(((kde_eces - gt_integral) / gt_integral) * 100)

    return diff_kde_ece_gt_integral, relative_bias, points


def plot_bias(points, diff_kde_ece_gt_integral, which_convex_fcn, num_classes, plot_name):
    mean_diff = np.mean(diff_kde_ece_gt_integral, axis=0)
    std_err = np.std(diff_kde_ece_gt_integral, axis=0) / np.sqrt(len(diff_kde_ece_gt_integral))
    plt.errorbar(points, mean_diff, fmt='o-', label=f'{num_classes} classes')
    plt.fill_between(points, mean_diff - std_err, mean_diff + std_err, alpha=0.25)

    y_label = f"CE ({which_convex_fcn.upper()}) bias" if plot_name == 'bias' \
        else f"Relative CE ({which_convex_fcn.upper()}) bias (%)"
    y_lim = (0, 0.1) if plot_name == 'bias' else (0, 250)
    plt.xlabel("Number of points")
    plt.ylabel(y_label)
    plt.ylim(y_lim)


def plot_bias_synthetic_data(which_convex_fcn, num_repeats=20):
    num_classes, num_samples, num_bins, temp1, temp2, _ = get_setup()

    for num_classes in [2, 5, 10]:
        diff_kde_ece_gt_integral, relative_bias, points = collect_results_for_nclasses(which_convex_fcn, num_classes, temp1, temp2, num_samples, num_repeats)
        plot_bias(points, diff_kde_ece_gt_integral, which_convex_fcn, num_classes, 'bias')
        # plot_bias(points, relative_bias, which_convex_fcn, num_classes, 'rel_bias')

    plt.legend()
    extra_params = [
        'scaled y ticks = false',
        'y tick label style={/pgf/number format/fixed}',
        'scaled x ticks = false',
        'x tick label style={/pgf/number format/fixed}',
        'xtick = {0, 5000, 10000, 15000, 20000}',
        '/pgf/number format/.cd, 1000 sep={}'
    ]
    tikzplotlib.clean_figure()
    tikzplotlib.save(
        os.path.join('figs', f'bias_convergence_synthetic_{which_convex_fcn}.tex'),
        extra_axis_parameters=extra_params,
        )
    plt.show()


def get_setup():
    num_classes = 2
    temp1 = 0.9
    temp2 = 0.6
    bandwidth = 0.001
    num_bins = 15
    num_samples = 5000

    return num_classes, num_samples, num_bins, temp1, temp2, bandwidth


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    n_repeats = 20

    compare_with_binning('kl', n_repeats)
    compare_two_impl('kl', n_repeats)
    plot_bias_synthetic_data('kl', n_repeats)

    # compare_with_binning('l2', n_repeats)
    # compare_two_impl('l2', n_repeats)
    # plot_ce('l2', 'canonical')
    # plot_ce('kl', 'canonical')
    # plot_ce('l2', 'classwise')
    # plot_ce('kl', 'classwise')
