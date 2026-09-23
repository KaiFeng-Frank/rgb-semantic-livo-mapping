#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distil_ext_rprime.py -- arm R-PRIME, added WITHOUT editing one byte of distil_ext.py,
random_supervise.py, filter_e.py, pointcept_ext.py or anything under
src/Pointcept_v151/pointcept/.

Import AFTER distil_ext (which itself must come after pointcept_ext).

WHY A SECOND MODULE INSTEAD OF A FIFTH `supervise` MODE IN distil_ext.py
-----------------------------------------------------------------------
Arms B1 / B0 / D / C / R were all trained against a distil_ext.py with a specific
md5 (03b70737662dc7d4c052a30fdaeeec37).  Arm R' is a LATER arm that must be comparable
to arm D, so the file every earlier arm ran must stay bit-identical on this machine.
Everything arm R' needs is therefore added by SUBCLASSING and by REGISTERING one new
transform.  The diff against the arms already measured is: one extra per-point array
emitted by the dataset, and one transform inserted after GridSample.

WHAT IS DIFFERENT FROM ARM R, IN ONE SENTENCE
---------------------------------------------
Arm R chooses its supervised POINTS before GridSample and therefore does not control
how many of them survive it; arm R' chooses its supervised VOXELS after GridSample and
therefore lands on arm D's post-grid count exactly.

THE TWO PIECES
--------------
DistilSemanticKITTIDatasetVR
    DistilSemanticKITTIDataset with supervise="voxel_random".  That mode name matches
    none of the base class's `if/elif` branches, so get_data() leaves `segment` as the
    FULL ground truth and `frustum` as the TRUE camera frustum -- byte-identical to
    supervise="all" -- which is exactly the pre-GridSample state arm R' needs, because
    its masking happens after the sample, not before it.  The subclass adds ONE array,
    `prio`, the frame's frozen per-point rank (see voxel_random_supervise.G2).

VoxelRandomSupervise
    Runs immediately AFTER GridSample, on one row per surviving voxel.  It computes arm
    D's post-grid supervised set in this very voxelisation, draws the same number of
    label-valid survivors by lowest rank, writes `segment` := GT on those and
    ignore_index elsewhere, and overloads `frustum` := the selection.

    THE `frustum` OVERLOAD IS DELIBERATE AND IS ARM R'S OWN PRECEDENT, NOT A NEW IDEA.
    DistilSegmentorMiB anchors the KL on `~frustum`, i.e. on whatever the dataset
    declares NOT supervised.  Arm R already reuses the key that way.  Writing the
    selection there is the only way to say "KL on the unsupervised voxels" without
    touching the model -- and touching the model would change arm D's own binary.
"""
import os, json, time

import numpy as np

import distil_ext as DX                                          # noqa: F401
from pointcept.datasets.builder import DATASETS                   # noqa: E402
from pointcept.datasets.transform import TRANSFORMS               # noqa: E402
from voxel_random_supervise import (frame_priority,               # noqa: E402
                                    voxel_random_select,
                                    VOXEL_RANDOM_SALT)


@DATASETS.register_module()
class DistilSemanticKITTIDatasetVR(DX.DistilSemanticKITTIDataset):
    """Arm R'.  Identical to the base class in every path it shares with arms C/D/R.

    The ONLY behavioural differences:
      * supervise="voxel_random" is accepted (the base class's assert lists four modes);
      * one extra per-point array, `prio`, is emitted so that the frame's frozen ranking
        survives GridSample together with coord / strength / segment / frustum.

    `segment` is emitted as the FULL ground truth and `frustum` as the TRUE camera
    frustum.  Neither is masked here: arm R' masks after the voxelisation, which is the
    whole point of the arm.
    """

    def __init__(self, **kw):
        kw = dict(kw)
        sup = kw.pop("supervise", None)
        assert sup == "voxel_random", \
            "DistilSemanticKITTIDatasetVR exists only for supervise='voxel_random'; " \
            "got %r -- use DistilSemanticKITTIDataset for the other modes" % (sup,)
        # The base class asserts supervise in ("all","frustum","filterE","random").
        # "all" is passed through the assert and then replaced: get_data() dispatches on
        # self.supervise, and "voxel_random" matches no branch, so it behaves exactly as
        # "all" does -- untouched GT segment, untouched camera frustum.  The replacement
        # is what makes the arm's real mode visible in logs and in repr().
        super().__init__(supervise="all", **kw)
        self.supervise = "voxel_random"
        self.salt = VOXEL_RANDOM_SALT
        print("[DistilSemanticKITTIDatasetVR] supervise=voxel_random  salt=%s  "
              "(segment and frustum leave get_data UNMASKED; VoxelRandomSupervise "
              "masks them after GridSample)" % self.salt, flush=True)

    def get_data(self, idx):
        d = super().get_data(idx)
        seq, frame = self._seq_frame(self.data_list[idx % len(self.data_list)])
        d["prio"] = frame_priority(seq, frame, d["segment"].shape[0], self.salt)
        d["seq_frame"] = "%s/%s" % (seq, frame)
        return d


@TRANSFORMS.register_module()
class VoxelRandomSupervise(object):
    """Arm R''s supervision selector, applied AFTER GridSample.

    Input  (one row per surviving voxel, straight out of GridSample):
        segment  int32[v]   FULL ground truth in common-9, ignore_index on excluded
        frustum  int32[v]   the TRUE camera frustum -- arm D's mask
        prio   float64[v]   the rank this voxel's representative point carries
    Output:
        segment  int32[v]   GT on the selected voxels, ignore_index everywhere else
        frustum  int32[v]   THE SELECTION.  DistilSegmentorMiB anchors its KL on the
                            complement of this key, so writing the selection here is
                            what makes the anti-forgetting term cover exactly the
                            non-supervised voxels -- the rule arm D obeys.
        prio, seq_frame     deleted; nothing downstream reads them.

    It draws no random number (voxel_random_supervise.G2).  `audit_path`, when set,
    appends one line per frame per worker; that is the only side effect.
    """

    def __init__(self, ignore_index=-1, audit_path=None):
        self.ignore_index = int(ignore_index)
        self.audit_path = audit_path
        self._fh = None

    def _audit(self, rec):
        if not self.audit_path:
            return
        if self._fh is None:
            # one file per DataLoader worker: num_worker=8 means eight processes run
            # this transform, and a shared handle would interleave partial lines.
            os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
            self._fh = open("%s.w%d" % (self.audit_path, os.getpid()), "a", buffering=1)
        self._fh.write(json.dumps(rec) + "\n")

    def __call__(self, data_dict):
        seg = data_dict["segment"]
        fru = np.asarray(data_dict["frustum"]).reshape(-1).astype(bool)
        prio = data_dict["prio"]
        valid = seg >= 0

        sel, k, n_pool = voxel_random_select(prio, fru, valid)

        self._audit(dict(f=data_dict.get("seq_frame", "?"),
                         v=int(seg.shape[0]),          # surviving voxels
                         pool=int(n_pool),             # label-valid survivors
                         k=int(k),                     # == arm D's post-grid count
                         sel=int(sel.sum()),           # == k, by construction
                         camfru=int(fru.sum()),        # camera-frustum survivors
                         t=round(time.time(), 3)))

        data_dict["segment"] = np.where(sel, seg, self.ignore_index).astype(np.int32)
        data_dict["frustum"] = sel.astype(np.int32)
        data_dict.pop("prio", None)
        data_dict.pop("seq_frame", None)
        return data_dict
