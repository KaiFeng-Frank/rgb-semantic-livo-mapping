"""Same-forward, bounded GPU pooling; only instance summaries cross to CPU."""
MAX_GROUPS = 128
FEATURE_DIM = 64


def pool_features(features, inverse, groups, count):
    import torch
    if not 0 <= count <= MAX_GROUPS or features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError('Invalid instance feature dimensions')
    g = torch.as_tensor(groups, device=features.device, dtype=torch.long)
    inv = torch.as_tensor(inverse, device=features.device, dtype=torch.long)
    if g.shape != inv.shape or torch.any((g < 0) | (g > count)):
        raise ValueError('Invalid instance feature groups')
    active = g > 0
    x = features[inv[active]].float()
    ids = g[active]-1
    total = x.new_zeros((count, FEATURE_DIM))
    square = total.clone()
    total.index_add_(0, ids, x)
    square.index_add_(0, ids, x.square())
    n = torch.bincount(ids, minlength=count).clamp_min(1).float()[:, None]
    mean = total/n
    std = (square/n-mean.square()).clamp_min(0).sqrt()
    return torch.cat((mean, std), dim=1)
