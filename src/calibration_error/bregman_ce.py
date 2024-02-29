import torch
from torch import nn


def get_bregman_ce(convex_fcn, gx, y, bandwidth, device='cpu'):
    """
    Calculate an estimate of Bregman calibration error.

    Args:
        convex_fcn: A strictly convex function F
        gx: The vector containing the probability scores, shape [num_samples, num_classes]
        y: The vector containing the labels, shape [num_samples]
        bandwidth: The bandwidth of the kernel
        device: The device type: 'cpu', 'cuda', 'mps'

    Returns: Bregman/proper calibration error characterized by the given convex function.

    """
    _check_input(gx)
    gx = _convert_to_device_and_dtype(gx, device)
    gx.detach()
    if not gx.requires_grad:
        gx.requires_grad = True

    ratio = _get_ratio(gx, y, bandwidth, device)
    ratio = torch.clamp(ratio, min=1e-45)
    f_ratio = convex_fcn(ratio)
    f_gx = convex_fcn(gx)
    grad_f_gx = torch.autograd.grad(f_gx, gx, grad_outputs=torch.ones_like(f_gx), retain_graph=True)[0]
    diff_ratio_gx = ratio - gx
    dot_prod = torch.sum(grad_f_gx * diff_ratio_gx, dim=1)
    CE = torch.mean(f_ratio - f_gx - dot_prod)

    _check_output(CE, 'CE')
    return CE


def get_bandwidth(gx, device):
    """
    Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).

    Args:
        gx: The vector containing the probability scores, shape [num_samples, num_classes]
        device: The device type: 'cpu' or 'cuda'

    Returns: The bandwidth of the kernel/
    """
    bandwidths = torch.cat((torch.logspace(start=-5, end=-1, steps=50), torch.linspace(0.2, 1, steps=5)))
    max_b = -1
    max_l = torch.finfo(torch.float).min
    n = len(gx)
    for b in bandwidths:
        log_kern = get_kernel(gx, b, device)
        log_fhat = torch.logsumexp(log_kern, 1) - torch.log(torch.tensor(n-1))
        l = torch.sum(log_fhat)
        if l > max_l:
            max_l = l
            max_b = b

    return max_b


def get_classwise_bregman_ce(convex_fcn, gx, y, bandwidth, device='cpu'):
    _check_input(gx)
    gx = _convert_to_device_and_dtype(gx, device)
    gx.detach()
    if not gx.requires_grad:
        gx.requires_grad = True

    all_CEf = []
    for i in range(gx.shape[1]):
        classwise_y = (y == i).long()
        gx_i = gx[:, i].unsqueeze(-1)
        classwise_gx = torch.cat((1 - gx_i, gx_i), dim=1)
        ratio = _get_ratio(classwise_gx, classwise_y, bandwidth, device)
        f_ratio = convex_fcn(ratio)
        f_gx = convex_fcn(classwise_gx)
        grad_f_gx = torch.autograd.grad(f_gx, classwise_gx, grad_outputs=torch.ones_like(f_gx), retain_graph=True)[0]
        diff_ratio_gx = ratio - classwise_gx
        dot_prod = torch.sum(grad_f_gx * diff_ratio_gx, dim=1)
        CEf = torch.mean(f_ratio - f_gx - dot_prod)
        all_CEf.append(CEf)

    all_CEf = torch.stack(all_CEf)

    _check_output(all_CEf, 'all_CEf')
    return torch.mean(all_CEf)


def get_bregman_ce_via_risk(convex_fcn, gx, y, bandwidth, device='cpu'):
    '''
    Calculate an estimate of Bregman calibration error using the formula derived via risk.

    Args:
        convex_fcn: A strictly convex function F
        gx: The vector containing the probability scores, shape [num_samples, num_classes]
        y: The vector containing the labels, shape [num_samples]
        bandwidth: The bandwidth of the kernel
        device: The device type: 'cpu', 'cuda', 'mps'

    Returns: Bregman calibration error characterized by the given convex function
    '''
    _check_input(gx)
    gx = _convert_to_device_and_dtype(gx, device)
    gx.detach()
    if not gx.requires_grad:
        gx.requires_grad = True

    ratio = _get_ratio(gx, y, bandwidth, device)
    ratio = torch.clamp(ratio, min=1e-45)
    f_ratio = convex_fcn(ratio)
    Sf = torch.mean(f_ratio)
    y_onehot = nn.functional.one_hot(y, num_classes=gx.shape[1])
    y_onehot = _convert_to_device_and_dtype(y_onehot, device)
    F_y = convex_fcn(y_onehot)
    Df = _get_bregman_divergence(convex_fcn, y_onehot, gx)
    CEf = Sf + torch.mean(Df - F_y)

    _check_output(CEf)
    return CEf


def get_bregman_stats_via_risk_kl(gx, y, bandwidth, device='cpu'):
    _check_input(gx)
    gx = _convert_to_device_and_dtype(gx, device)

    ratio = _get_ratio(gx, y, bandwidth, device)
    f_ratio = negative_entropy(ratio)
    Sf = torch.mean(f_ratio)
    y_onehot = nn.functional.one_hot(y, num_classes=gx.shape[1])
    y_onehot = _convert_to_device_and_dtype(y_onehot, device)
    F_y = negative_entropy(y_onehot)
    Df = nn.KLDivLoss(reduction='none')(torch.log(gx), y_onehot).sum(dim=1)
    CEf = Sf + torch.mean(Df - F_y)
    E_Y = torch.mean(y_onehot, dim=0)
    F_E_Y = torch.sum(E_Y * torch.log(E_Y))
    sharpness = Sf - F_E_Y

    _check_output(CEf, 'CEf')
    _check_output(Sf, 'Sf')
    _check_output(sharpness, 'sharpness')
    return CEf, Sf, sharpness


def get_bregman_stats_via_risk_l2(gx, y, bandwidth, device='cpu'):
    _check_input(gx)
    gx = _convert_to_device_and_dtype(gx, device)

    ratio = _get_ratio(gx, y, bandwidth, device)
    f_ratio = l2_norm(ratio)
    Sf = torch.mean(f_ratio)
    y_onehot = nn.functional.one_hot(y, num_classes=gx.shape[1])
    y_onehot = _convert_to_device_and_dtype(y_onehot, device)
    F_y = l2_norm(y_onehot)
    Df = torch.sum(torch.square(y_onehot - gx), dim=1)
    CEf = Sf + torch.mean(Df - F_y)
    E_Y = torch.mean(y_onehot, dim=0)
    F_E_Y = torch.sum(torch.pow(E_Y, 2))
    sharpness = Sf - F_E_Y

    _check_output(CEf, 'CEf')
    _check_output(Sf, 'Sf')
    _check_output(sharpness, 'sharpness')
    return CEf, Sf, sharpness


def get_classwise_bregman_stats_via_risk_kl(gx, y, bandwidth, device='cpu'):
    _check_input(gx)
    gx = _convert_to_device_and_dtype(gx, device)

    all_CEf = []
    all_Sf = []
    all_sharpness = []
    for i in range(gx.shape[1]):
        classwise_y = (y == i).long()
        gx_i = gx[:, i].unsqueeze(-1)
        classwise_gx = torch.cat((1 - gx_i, gx_i), dim=1)
        ratio = _get_ratio(classwise_gx, classwise_y, bandwidth, device)
        f_ratio = negative_entropy(ratio)
        Sf = torch.mean(f_ratio)
        classwise_y_onehot = nn.functional.one_hot(classwise_y, num_classes=classwise_gx.shape[1])
        classwise_y_onehot = _convert_to_device_and_dtype(classwise_y_onehot, device)
        F_y = negative_entropy(classwise_y_onehot)
        Df = nn.KLDivLoss(reduction='none')(torch.log(classwise_gx), classwise_y_onehot).sum(dim=1)
        CEf = Sf + torch.mean(Df - F_y)
        E_Y = torch.mean(classwise_y_onehot, dim=0)
        F_E_Y = torch.sum(E_Y * torch.log(E_Y))
        sharpness = Sf - F_E_Y
        # Refinement in the paper is defined as negative
        all_Sf.append(-Sf)
        all_CEf.append(CEf)
        all_sharpness.append(sharpness)

    all_CEf = torch.stack(all_CEf)
    all_Sf = torch.stack(all_Sf)
    all_sharpness = torch.stack(all_sharpness)

    _check_output(all_CEf, 'all_CEf')
    _check_output(all_Sf, 'all_Sf')
    _check_output(all_sharpness, 'all_sharpness')
    return torch.mean(all_CEf), torch.mean(all_Sf), torch.mean(all_sharpness)


def get_classwise_bregman_stats_via_risk_l2(gx, y, bandwidth, device='cpu'):
    _check_input(gx)
    gx = _convert_to_device_and_dtype(gx, device)

    all_CEf = []
    all_Sf = []
    all_sharpness = []
    for i in range(gx.shape[1]):
        classwise_y = (y == i).long()
        gx_i = gx[:, i].unsqueeze(-1)
        classwise_gx = torch.cat((1 - gx_i, gx_i), dim=1)
        ratio = _get_ratio(classwise_gx, classwise_y, bandwidth, device)
        f_ratio = l2_norm(ratio)
        Sf = torch.mean(f_ratio)
        classwise_y_onehot = nn.functional.one_hot(classwise_y, num_classes=classwise_gx.shape[1])
        classwise_y_onehot = _convert_to_device_and_dtype(classwise_y_onehot, device)
        F_y = l2_norm(classwise_y_onehot)
        Df = torch.sum(torch.square(classwise_y_onehot - classwise_gx), dim=1)
        CEf = Sf + torch.mean(Df - F_y)
        E_Y = torch.mean(classwise_y_onehot, dim=0)
        F_E_Y = torch.sum(torch.pow(E_Y, 2))
        sharpness = Sf - F_E_Y
        # Refinement in the paper is defined as negative
        all_Sf.append(-Sf)
        all_CEf.append(CEf)
        all_sharpness.append(sharpness)

    all_CEf = torch.stack(all_CEf)
    all_Sf = torch.stack(all_Sf)
    all_sharpness = torch.stack(all_sharpness)

    _check_output(all_CEf, 'all_CEf')
    _check_output(all_Sf, 'all_Sf')
    _check_output(all_sharpness, 'all_sharpness')
    return torch.mean(all_CEf), torch.mean(all_Sf), torch.mean(all_sharpness)


def _get_ratio(f, y, bandwidth, device='cpu'):
    if f.shape[1] > 20:
        # Slower but more numerically stable implementation for larger number of classes
        return _get_ratio_iter(f, y, bandwidth, device)

    log_kern = get_kernel(f, bandwidth, device)

    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1])
    y_onehot = _convert_to_device_and_dtype(y_onehot, device)
    # matrix multiplication in log space using broadcasting
    log_kern_y = torch.logsumexp(log_kern.unsqueeze(2) + torch.log(y_onehot).unsqueeze(0), dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_kern_y - log_den.unsqueeze(-1)
    ratio = torch.exp(log_ratio)

    _check_output(ratio, 'ratio')
    return ratio


def _get_ratio_iter(f, y, bandwidth, device='cpu'):
    log_kern = get_kernel(f, bandwidth, device)
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1])
    y_onehot = _convert_to_device_and_dtype(y_onehot, device)
    log_y = torch.log(y_onehot)
    log_den = torch.logsumexp(log_kern, dim=1)
    final_ratio = []
    for k in range(f.shape[1]):
        log_kern_y = log_kern + (torch.ones([f.shape[0], 1]).to(device) * log_y[:, k].unsqueeze(0))
        log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
        inner_ratio = torch.exp(log_inner_ratio)
        final_ratio.append(inner_ratio)

    return torch.transpose(torch.stack(final_ratio), 0, 1)


def get_kernel(f, bandwidth, device):
    # if num_classes == 1
    if f.shape[1] == 1:
        log_kern = _beta_kernel(f, f, bandwidth).squeeze()
    else:
        log_kern = _dirichlet_kernel(f, bandwidth).squeeze()
    # Trick: -inf on the diagonal
    return log_kern + torch.diag(torch.finfo(torch.double).min * torch.ones(len(f))).to(device)


def _beta_kernel(z, zi, bandwidth=0.1):
    p = zi / bandwidth + 1
    q = (1-zi) / bandwidth + 1
    z = z.unsqueeze(-2)

    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    log_num = (p-1) * torch.log(z) + (q-1) * torch.log(1-z)
    log_beta_pdf = log_num - log_beta

    return log_beta_pdf


def _dirichlet_kernel(z, bandwidth=0.1):
    alphas = z / bandwidth + 1

    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
    log_num = torch.matmul(torch.log(z), (alphas-1).T)
    log_dir_pdf = log_num - log_beta

    return log_dir_pdf


def _get_bregman_divergence(convex_fcn, p, q):
    q = q.detach()
    if not q.requires_grad:
        q.requires_grad = True

    f_p = convex_fcn(p)
    f_q = convex_fcn(q)
    grad_f_q = torch.autograd.grad(f_q, q, grad_outputs=torch.ones_like(f_q), retain_graph=True)[0]
    dot_prod = torch.sum(grad_f_q * (p - q), dim=1)

    return f_p - f_q - dot_prod


def _check_input(gx):
    assert not _isnan(gx), "input contains nans"
    assert len(gx.shape) == 2
    assert torch.min(gx) >= 0
    assert torch.max(gx) <= 1


def _convert_to_device_and_dtype(x, device):
    # mps does not support double
    if device == 'mps':
        return x.float().to(device)
    # for cpu and cuda, convert to double precision
    else:
        if x.dtype != torch.double:
            x = x.double().to(device)
    return x


def _check_output(out, name=""):
    assert not _isnan(out), f"{name} contains nans"


def _isnan(a):
    return torch.any(torch.isnan(a))


def l2_norm(x):
    return torch.sum(torch.pow(x, 2), dim=1)


def negative_entropy(x):
    neg_entropy = torch.sum(x * torch.log(x), dim=1)
    neg_entropy[torch.isnan(neg_entropy)] = torch.tensor(0)

    return neg_entropy
