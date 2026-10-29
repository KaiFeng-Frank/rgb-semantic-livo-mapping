"""Equivalent one-pass bounds for the existing discrete detector index image."""
import numpy as np
from scipy.ndimage import find_objects


def detector_boxes(index, count):
    """Same min/max+1 boxes as per-instance np.nonzero, including fragments.

    Input must satisfy InstanceFrame's validated index invariant: zero is
    unassigned, positive labels are slots 1..count. Empty slots stay absent.
    This does not alter masks, erosion, identities, or prompts.
    """
    if count == 0:
        return [], np.empty((0, 4), np.float32)
    regions = find_objects(index, max_label=count)
    slots = [s for s, region in enumerate(regions) if region is not None]
    boxes = [[regions[s][1].start, regions[s][0].start,
              regions[s][1].stop, regions[s][0].stop] for s in slots]
    return slots, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
