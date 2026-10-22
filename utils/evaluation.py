import torch


def mass_metrics(prediction, target):
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    diff = prediction - target
    diff_log = torch.log(prediction) - torch.log(target)
    ratio = prediction / target
    inverse_ratio = target / prediction

    return {
        'alde': torch.mean(torch.abs(diff_log)).item(),
        'ape': torch.mean(torch.abs(diff / target)).item(),
        'mnre': torch.mean(torch.minimum(ratio, inverse_ratio)).item(),
        'rmse_log': torch.sqrt(torch.mean(diff_log.pow(2))).item(),
        'log10': torch.mean(torch.abs(torch.log10(prediction) - torch.log10(target))).item(),
        'silog': torch.sqrt(
            torch.clamp(diff_log.pow(2).mean() - 0.5 * diff_log.mean().pow(2), min=0)
        ).item(),
        'q': torch.mean((torch.maximum(ratio, inverse_ratio) < 2.0).float()).item(),
        'signed_log_bias': diff_log.mean().item(),
    }


def composition_metrics(composition, target, material_densities, fill_factor=None):
    target = target.to(dtype=composition.dtype)
    labels = target > 0
    valid = labels.any(dim=1)
    if not valid.any():
        return {}

    comp = composition[valid]
    target = target[valid]
    labels = labels[valid]
    out = {}

    for k in (1, 3, 5):
        topk = comp.topk(k=min(k, comp.shape[1]), dim=1).indices
        out['recall_at_%d' % k] = (target.gather(1, topk).sum(dim=1) > 0).float().mean().item()

    labelled_fraction = (comp * target).sum(dim=1)
    entropy = -(comp * torch.log(comp.clamp_min(1e-8))).sum(dim=1)
    max_fraction, top1 = comp.max(dim=1)

    densities = material_densities.to(dtype=comp.dtype, device=comp.device).reshape(1, -1)
    raw_mass = comp * densities
    mass_fraction = raw_mass / raw_mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
    labelled_mass_fraction = (mass_fraction * target).sum(dim=1)
    mass_top1 = raw_mass.argmax(dim=1)

    neg_inf = torch.finfo(comp.dtype).min
    labelled_max = comp.masked_fill(~labels, neg_inf).max(dim=1).values
    has_unlabelled = (~labels).any(dim=1)
    unlabelled_max = comp.masked_fill(labels, neg_inf).max(dim=1).values
    unlabelled_max = torch.where(has_unlabelled, unlabelled_max, torch.zeros_like(unlabelled_max))

    out.update(
        {
            'labelled_fraction': labelled_fraction.mean().item(),
            'mass_recall_at_1': target.gather(1, mass_top1.unsqueeze(1)).squeeze(1).gt(0).float().mean().item(),
            'labelled_mass_fraction': labelled_mass_fraction.mean().item(),
            'dominance_margin': (labelled_max - unlabelled_max).mean().item(),
            'max_fraction': max_fraction.mean().item(),
            'entropy': entropy.mean().item(),
            'effective_materials': torch.exp(entropy).mean().item(),
        }
    )

    if fill_factor is not None and fill_factor.numel():
        fill = fill_factor.reshape(-1).float()
        out['fill_mean'] = fill.mean().item()
        out['fill_p50'] = torch.quantile(fill, 0.50).item()
    return out
