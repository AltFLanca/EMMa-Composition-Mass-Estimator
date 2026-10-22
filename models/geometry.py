import torch


def _valid_xyz(pc):
    if pc.ndim == 4 and pc.shape[1] == 1 and pc.shape[-1] == 3:
        xyz = pc[:, 0]
        layout = 'b1n3'
    elif pc.ndim == 3 and pc.shape[-1] == 3:
        xyz = pc
        layout = 'bn3'
    else:
        raise ValueError('Expected point cloud [B,1,N,3] or [B,N,3], got %s' % (tuple(pc.shape),))

    valid = xyz.abs().sum(dim=-1) > 1e-12
    return xyz, layout, valid


def robust_rms_normalize(pc, input_divisor, trim_quantile=0.95, eps=1e-8):
    xyz, layout, valid = _valid_xyz(pc)

    xyz_nan = xyz.masked_fill(~valid.unsqueeze(-1), float('nan'))
    centre = torch.nanmedian(xyz_nan, dim=1).values
    centre = torch.nan_to_num(centre, nan=0.0)

    centered = xyz - centre.unsqueeze(1)
    centered = torch.where(valid.unsqueeze(-1), centered, torch.zeros_like(centered))

    radii = torch.linalg.vector_norm(centered, dim=-1)
    radii_nan = radii.masked_fill(~valid, float('nan'))
    cutoff = torch.nanquantile(radii_nan, trim_quantile, dim=1, keepdim=True)
    cutoff = torch.nan_to_num(cutoff, nan=0.0).clamp_min(eps)

    clipped = torch.minimum(radii, cutoff)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(xyz.dtype)
    rms = torch.sqrt((clipped.square() * valid).sum(dim=1, keepdim=True) / count)
    rms = rms.clamp_min(eps)

    normalized = centered / rms.unsqueeze(-1)
    normalized = torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    if layout == 'b1n3':
        normalized = normalized.unsqueeze(1)

    return normalized, rms * float(input_divisor)
