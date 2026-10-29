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


def current_token_input(joint,feature_dim=64,classes=16):
    """Exact existing current-token encoding; no training labels accepted."""
    import torch
    if joint.ndim!=2 or joint.shape[1]!=3*feature_dim+2*classes+5:
        raise ValueError('Invalid JointRepair input shape')
    z=torch.zeros_like(joint)
    z[:,:feature_dim]=joint[:,:feature_dim]
    z[:,feature_dim:2*feature_dim]=joint[:,:feature_dim]
    z[:,3*feature_dim:3*feature_dim+classes]=joint[:,3*feature_dim:3*feature_dim+classes]
    z[:,-1]=1
    return z


def supervision_groups(anchor,labels,historical_labels,valid,residual_limit=2.):
    """Training-only masks: repair=0, preserve=1, other=2, ignored=-1."""
    import torch
    y=labels.long();good=y>=0;correct=anchor.argmax(1)==y
    helpful=(valid&(historical_labels==y[:,None])).any(1)
    conflict=(valid&(historical_labels!=y[:,None])).any(1)
    gap=anchor.max(1).values-anchor.gather(1,y.clamp_min(0)[:,None]).squeeze(1)
    group=torch.full_like(y,2)
    group[good&~correct&helpful&(gap<2*residual_limit)]=0
    group[good&correct&conflict]=1
    group[~good]=-1
    return group


def repair_objective(logits,anchor,labels,groups,weights,balanced=False):
    """Identical existing point loss; only its group reduction may differ."""
    import torch
    from torch.nn import functional as F
    valid=labels>=0
    if not bool(valid.any()):return logits.sum()*0,logits.new_zeros(3)
    y=labels[valid].long();z=logits[valid];base=anchor[valid].detach();g=groups[valid]
    if bool(((g<0)|(g>2)).any()):raise ValueError('Every supervised point needs a group')
    w=weights[y]
    if bool((w<=0).any()):raise ValueError('Observed training targets need positive class weights')
    mask=F.one_hot(y,z.shape[1]).bool()
    margin=z.gather(1,y[:,None]).squeeze(1)-z.masked_fill(mask,-1e4).max(1).values
    old_margin=base.gather(1,y[:,None]).squeeze(1)-base.masked_fill(mask,-1e4).max(1).values
    correct=base.argmax(1)==y
    loss=w*(F.cross_entropy(z,y,reduction='none')+.5*(~correct)*F.relu(.5-margin)+.5*correct*F.relu(old_margin.clamp(max=2)-margin))
    mass=torch.stack([loss[g==i].sum() for i in range(3)])
    denom=torch.stack([w[g==i].sum() for i in range(3)])
    if balanced:
        exists=denom>0
        return (mass[exists]/denom[exists]).mean(),mass.detach()
    return loss.sum()/w.sum(),mass.detach()
