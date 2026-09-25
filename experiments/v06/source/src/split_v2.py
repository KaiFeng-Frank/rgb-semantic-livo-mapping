"""v0.6 split: seq 09 joins seq 07 as a held-out TEST sequence.

WHY THIS FILE EXISTS
--------------------
Arms B1 / D / B0 / C / R / Rprime were trained under

    SPLIT2SEQ = train[0,1,2,4,5,6,9,10]  val[8]  test[7]

seq 07 never entered training and never selected a checkpoint.  But every DESIGN
decision of v0.4 / v0.5 -- whether to keep the anti-forgetting KL anchor, the
confidence-gate threshold, the 0.20 m map voxel -- was made while looking at seq 07
numbers.  That is DECISION contamination: weaker than training contamination, invisible
to a train/test audit, and it biases the final number optimistically.  seq 09 is
withheld here so that one number exists which no design decision has ever seen.

seq 10 STAYS IN TRAIN, deliberately.  Withdrawing 09 and 10 together costs 16% of the
training frames; withdrawing 09 alone costs 7%, which keeps the new absolute numbers
comparable to the v0.4 / v0.5 ones.  seq 09 is 1591 frames on its own -- larger than
seq 07's 1101 -- so it carries enough statistics as a single held-out sequence.
Variance is supplied by the three seeds, not by a second held-out sequence.

seq 03 is absent for a DATA reason, unchanged from pointcept_ext.py: its KITTI raw
drive 2011_09_26_drive_0067 returns HTTP 404 (re-verified 2026-09-24).

pointcept_ext.py, distil_ext.py and distil_ext_rprime.py are NOT touched by this file,
so every v0.4 / v0.5 arm stays reproducible from its original config at its original
md5.  This module only subclasses.
"""
import os

from pointcept.datasets.builder import DATASETS

import pointcept_ext as PX
import distil_ext as DX
import distil_ext_rprime as RX

SPLIT2SEQ_V2 = dict(
    train=[0, 1, 2, 4, 5, 6, 10],
    val=[8],
    test=[7, 9],
)

HELD_OUT = frozenset({7, 9})

# Authoritative for v0.6.  src/seqreg.py ALSO carries split constants -- TRAIN_SEQS with
# "09" in it and TEST_SEQS == ["07"] -- and those are the v0.4/v0.5 split.  They are left
# untouched so the old arms stay reproducible, which means two sources of truth now exist.
# Anything scoring v0.6 must read these names, not seqreg's: taking seqreg.TEST_SEQS would
# silently skip seq 09, i.e. skip the only sequence no design decision has seen.
TRAIN_SEQS_V2 = ["%02d" % s for s in SPLIT2SEQ_V2["train"]]
VAL_SEQS_V2 = ["%02d" % s for s in SPLIT2SEQ_V2["val"]]
TEST_SEQS_V2 = ["%02d" % s for s in SPLIT2SEQ_V2["test"]]


def _static_check(split2seq):
    """Fails at import time, not at epoch 1, if the table itself leaks."""
    leak = (set(split2seq["train"]) | set(split2seq["val"])) & HELD_OUT
    assert not leak, (
        "held-out sequence(s) %s appear in train/val of SPLIT2SEQ_V2" % sorted(leak))
    assert set(split2seq["test"]) == HELD_OUT, (
        "test must be exactly the held-out set %s, got %s"
        % (sorted(HELD_OUT), sorted(split2seq["test"])))


_static_check(SPLIT2SEQ_V2)


def _is_test(split):
    if isinstance(split, str):
        return split == "test"
    return "test" in tuple(split)


def _guard(split, data_list):
    """Bit-level guard: look at the paths actually loaded, not at the table.

    A subclass that forgets to override SPLIT2SEQ, a stale .pyc, or a config that names
    the v0.4 dataset class would all pass the static check above and still train on
    seq 09.  This reads the sequence id back out of every path that was loaded.
    """
    if _is_test(split):
        return
    seen = set()
    for p in data_list:
        parts = p.replace(os.sep, "/").split("/sequences/")
        if len(parts) > 1:
            seen.add(parts[1][:2])
    leak = sorted(seen & {"%02d" % s for s in HELD_OUT})
    if leak:
        n = sum(1 for p in data_list
                if any("/sequences/%s/" % s in p.replace(os.sep, "/") for s in leak))
        raise AssertionError(
            "held-out sequence(s) %s leaked into split %r -- %d frames loaded from them"
            % (leak, split, n))


@DATASETS.register_module()
class SemanticKITTICommon9DatasetV2(PX.SemanticKITTICommon9Dataset):
    """v0.4 base dataset, v0.6 split."""

    SPLIT2SEQ = SPLIT2SEQ_V2

    def get_data_list(self):
        data_list = super().get_data_list()
        _guard(self.split, data_list)
        return data_list


@DATASETS.register_module()
class DistilSemanticKITTIDatasetV2(DX.DistilSemanticKITTIDataset):
    """Arms B0 / B0_noKL, v0.6 split.  Rare-class resampling path is inherited."""

    SPLIT2SEQ = SPLIT2SEQ_V2

    def get_data_list(self):
        data_list = super().get_data_list()
        _guard(self.split, data_list)
        return data_list


@DATASETS.register_module()
class DistilSemanticKITTIDatasetVRV2(RX.DistilSemanticKITTIDatasetVR):
    """Arm Rprime (upper bound), v0.6 split."""

    SPLIT2SEQ = SPLIT2SEQ_V2

    def get_data_list(self):
        data_list = super().get_data_list()
        _guard(self.split, data_list)
        return data_list


def assert_not_seqreg_split(seqs, what="scoring"):
    """Call this from anything that scores v0.6.  Cheap, and it fires on the one mistake
    that would be invisible in the output: scoring only seq 07 because seqreg said so."""
    seqs = ["%02d" % s if isinstance(s, int) else str(s) for s in seqs]
    missing = [s for s in TEST_SEQS_V2 if s not in seqs]
    if missing:
        raise AssertionError(
            "v0.6 %s is missing held-out sequence(s) %s -- got %s.  If this came from "
            "seqreg.TEST_SEQS, that constant is the v0.4/v0.5 split; use TEST_SEQS_V2."
            % (what, missing, seqs))
