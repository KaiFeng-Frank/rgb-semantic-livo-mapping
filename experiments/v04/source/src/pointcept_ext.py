#!/usr/bin/env python3
"""
pointcept_ext.py -- everything Pointcept v1.5.1 needs added to train PTv3 on OUR data,
without editing a single file under src/Pointcept_v151/pointcept/.

Import this module BEFORE building any dataset/model. It:
  1. installs the pointops / pointgroup_ops stubs (pointcept.models.__init__ and
     pointcept.engines.hooks.evaluator both import pointops at module scope),
  2. registers SemanticKITTICommon9Dataset.

WHY A NEW DATASET CLASS AND NOT THE STOCK ONE
---------------------------------------------
Three independent reasons, each of which alone is disqualifying:

  (a) TEST-SET LEAK.  pointcept/datasets/semantic_kitti.py:41 has
          train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
      i.e. the stock loader TRAINS ON SEQ 07, which is this project's held-out test
      sequence.  Using it unchanged silently invalidates every number downstream.

  (b) INTENSITY SCALE.  The stock loader does `strength = scan[:, -1]` with no scaling
      (NuScenesDataset divides by 255 because its .bin holds 0-255).  Both end up
      nominally in [0,1], but the DISTRIBUTIONS differ by ~5x: KITTI mean 0.293 vs
      nuScenes-after-/255 ~0.06.  The verified inference path multiplies KITTI
      intensity by 0.2 and that was measured, not guessed (CRITICAL_CONSTRAINTS C3:
      point acc 56.5 -> 87.6 going from x1.0 to x0.2).  Training must use the SAME
      scale the deployed model is fed, or train and test see different sensors.

  (c) LABEL SPACE.  We score in COMMON-9, not SemanticKITTI-19.

SUPERVISION MASKS
-----------------
Arms B and D supervise only the camera-visible points.  `mask_root` points at a
directory of <seq>/<frame>.npy boolean arrays (True = supervised).  Points with
mask False get ignore_index, so they are in the input but contribute no loss --
which is exactly the question v0.4 asks.

PSEUDO-LABELS
-------------
Arm B replaces GT with the 2D teacher's common-9 labels.  `label_root` points at a
directory of <seq>/<frame>.npy int8 arrays in COMMON-9 ids, with ignore_index where
the teacher abstains.  When label_root is set the .label files are not read at all.
"""
import os, sys, types
import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
POINTCEPT_V151 = "/data/wuyou/livo_sem/src/Pointcept_v151"


# --------------------------------------------------------------------------- #
# 1. stubs -- identical contract to ptv3_loader_verified._install_stubs
# --------------------------------------------------------------------------- #
class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self


class _Stub(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return _Any


def install_stubs():
    for n in ("pointops", "pointgroup_ops"):
        if n not in sys.modules:
            m = _Stub(n); m.__file__ = f"<stub {n}>"; sys.modules[n] = m


install_stubs()


# --------------------------------------------------------------------------- #
# 1b. yapf shim.  pointcept/utils/config.py:496 (Config.pretty_text, used by
#     default_config_parser -> cfg.dump) calls
#         FormatCode(text, style_config=..., verify=True)
#     yapf dropped `verify` in 0.40; this env has 0.43, so EVERY training launch
#     dies with "FormatCode() got an unexpected keyword argument 'verify'" before
#     a single sample is read.  Patch the callee, not the pinned repo.
# --------------------------------------------------------------------------- #
def _patch_yapf():
    try:
        from yapf.yapflib import yapf_api
    except Exception:
        return
    orig = yapf_api.FormatCode
    try:
        import inspect
        if "verify" in inspect.signature(orig).parameters:
            return
    except (TypeError, ValueError):
        return

    def FormatCode(*a, **k):
        k.pop("verify", None)
        out = orig(*a, **k)
        return out if isinstance(out, tuple) else (out, False)

    yapf_api.FormatCode = FormatCode
    import pointcept.utils.config as _pc_cfg  # noqa: E402  (may not be imported yet)
    if hasattr(_pc_cfg, "FormatCode"):
        _pc_cfg.FormatCode = FormatCode


if POINTCEPT_V151 not in sys.path:
    sys.path.insert(0, POINTCEPT_V151)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pointcept.datasets.builder import DATASETS          # noqa: E402
from pointcept.datasets.defaults import DefaultDataset   # noqa: E402
import label_spaces as LS                                # noqa: E402

_patch_yapf()


# --------------------------------------------------------------------------- #
# 2. COMMON-9 names, in the order label_spaces.COARSE defines
# --------------------------------------------------------------------------- #
COMMON9_NAMES = list(LS.COARSE)
NUM_COMMON9 = len(COMMON9_NAMES)

# SemanticKITTI raw id -> common-9, EXCLUDED(-1) folded onto ignore_index by the loader
SK_RAW_TO_COMMON9 = LS.sk_lut()          # int8[300], -1 == EXCLUDED
# nuScenes-16 index -> common-9, UNMAPPED(-2) for other_flat
NUSC16_TO_COMMON9 = LS.nusc_lut()        # int8[16]

# which nuScenes-16 indices form each common-9 group (used by the grouped-CE option
# that keeps the pretrained 16-way classifier -- see the report)
COMMON9_TO_NUSC16 = [
    [i for i, c in enumerate(NUSC16_TO_COMMON9) if c == g] for g in range(NUM_COMMON9)
]

STRENGTH_SCALE_KITTI = 0.2               # MEASURED, see CRITICAL_CONSTRAINTS C3


@DATASETS.register_module()
class SemanticKITTICommon9Dataset(DefaultDataset):
    """SemanticKITTI in COMMON-9, KITTI-raw-backed, with 07 held out of train."""

    # 07 is the TEST sequence for this project. It is NOT in train.
    #
    # 03 is NOT in train either, and that is a DATA fact, not a choice: SemanticKITTI
    # seq 03 maps to KITTI raw drive 2011_09_26_drive_0067, which returns HTTP 404 from
    # the KITTI mirror (verified 2026-09-22 with curl -I). The 801 .label files exist but
    # there are no points and no images, so the frames cannot be built at all. The train
    # split is therefore 8 sequences / 17,228 frames, not 9 / 18,029.
    SPLIT2SEQ = dict(
        train=[0, 1, 2, 4, 5, 6, 9, 10],
        val=[8],
        test=[7],
    )

    def __init__(
        self,
        split="train",
        data_root="/data/wuyou/livo_sem/data/pointcept_sk",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        strength_scale=STRENGTH_SCALE_KITTI,
        mask_root=None,      # arms B/D: only these points carry loss
        label_root=None,     # arm B: pseudo-labels instead of GT
    ):
        self.ignore_index = ignore_index
        self.strength_scale = float(strength_scale)
        self.mask_root = mask_root
        self.label_root = label_root
        lut = np.asarray(SK_RAW_TO_COMMON9, dtype=np.int32).copy()
        lut[lut < 0] = ignore_index       # EXCLUDED -> ignore
        self.raw_lut = lut
        super().__init__(
            split=split, data_root=data_root, transform=transform,
            test_mode=test_mode, test_cfg=test_cfg, loop=loop,
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            seq_list = self.SPLIT2SEQ[self.split]
        elif isinstance(self.split, (list, tuple)):
            seq_list = [s for sp in self.split for s in self.SPLIT2SEQ[sp]]
        else:
            raise NotImplementedError(self.split)
        assert 7 not in seq_list or self.split == "test" or "test" in self.split, \
            "seq 07 is the held-out TEST sequence; it must never appear in train/val"
        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            folder = os.path.join(self.data_root, "dataset", "sequences", seq, "velodyne")
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"{folder} missing -- run tools/make_sk_layout.py for seq {seq}")
            data_list += [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
        return data_list

    @staticmethod
    def _seq_frame(path):
        frame = os.path.splitext(os.path.basename(path))[0]
        seq = os.path.basename(os.path.dirname(os.path.dirname(path)))
        return seq, frame

    def get_data(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = (scan[:, 3] * self.strength_scale).reshape(-1, 1)
        seq, frame = self._seq_frame(path)

        if self.label_root is not None:                       # arm B: pseudo-labels
            p = os.path.join(self.label_root, seq, frame + ".npy")
            segment = np.load(p).astype(np.int32).reshape(-1)
            segment[segment < 0] = self.ignore_index
        else:                                                  # arms C/D: GT
            lf = os.path.join(os.path.dirname(os.path.dirname(path)),
                              "labels", frame + ".label")
            raw = np.fromfile(lf, dtype=np.uint32) & 0xFFFF
            segment = self.raw_lut[raw].astype(np.int32)

        if self.mask_root is not None:                         # arms B/D: sparsify
            m = np.load(os.path.join(self.mask_root, seq, frame + ".npy"))
            segment = np.where(m.reshape(-1), segment, self.ignore_index).astype(np.int32)

        assert segment.shape[0] == coord.shape[0], (
            f"{seq}/{frame}: {coord.shape[0]} points vs {segment.shape[0]} labels")
        return dict(coord=coord, strength=strength, segment=segment)

    def get_data_name(self, idx):
        seq, frame = self._seq_frame(self.data_list[idx % len(self.data_list)])
        return f"{seq}_{frame}"
