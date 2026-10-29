"""Training-only weighting helpers; absent classes must not set the normalization."""
import numpy as np


def present_class_weights(counts):
    counts=np.asarray(counts,np.float64)
    if counts.ndim!=1 or not len(counts) or not np.isfinite(counts).all() or (counts<0).any() or not (counts>0).any():
        raise ValueError('Nonnegative class counts with at least one observed class required')
    present=counts>0
    inverse=np.sqrt(counts.sum()/counts[present])
    weights=np.zeros_like(counts)
    weights[present]=np.clip(inverse/inverse.mean(),.25,4.)
    return weights
