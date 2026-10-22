import torch


def composition_losses(composition, material_target, target_mode='presence', hierarchy=None):
    """Material-presence/frontier loss plus entropy regularizer."""
    if target_mode == 'frontier':
        if hierarchy is None:
            raise ValueError('frontier material supervision requires a hierarchy')
        target = hierarchy.frontier(material_target)
    elif target_mode == 'presence':
        target = material_target
    else:
        raise ValueError('Unknown material target mode: %s' % target_mode)

    target = target.to(dtype=composition.dtype)
    labelled_fraction = (composition * target).sum(dim=1)
    presence = -torch.log(labelled_fraction.clamp_min(1e-8)).mean()
    entropy = -(composition * torch.log(composition.clamp_min(1e-8))).sum(dim=1).mean()
    return presence, entropy
