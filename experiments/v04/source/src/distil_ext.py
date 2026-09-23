#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distil_ext.py -- the v0.4 frozen design, implemented on Pointcept v1.5.1 without
editing one byte under src/Pointcept_v151/pointcept/.

Import AFTER pointcept_ext (which installs the pointops/pointgroup_ops stubs and
registers SemanticKITTICommon9Dataset).

WHAT THE FROZEN DESIGN SAYS, AND WHERE IT IS HERE
-------------------------------------------------
HEAD.  "keep the original 16-way nuScenes head.  Do NOT build a 9-way head.  Compute
the supervised loss by MiB-style marginalisation p9(c) = sum over k in g(c) of p16(k)."
  -> DistilSegmentorMiB.  seg_head stays nn.Linear(64, 16) and is loaded from the
     released checkpoint, so the model is the strict-488/488 object end to end and has
     ZERO new parameters.  nuScenes index 11 (other_flat) belongs to NO common-9 group,
     so sum_c p9(c) < 1 and the model is penalised for parking mass there -- which is
     what keeps the UNMAPPED accounting of the scorer meaningful.  Do not renormalise.

ANTI-FORGETTING.  "for EVERY out-of-frustum point add a KL to the FROZEN original PTv3,
computed on the FULL 16-dim simplex.  Frozen model runs no-grad."
  -> self.frozen (a second PTv3 + head, requires_grad_(False), .eval()), fed the SAME
     augmented cloud, KL(p_frozen || p_student) averaged over out-of-frustum points.

LAMBDA.  "Calibrate lambda so that at step 0 this term's gradient norm matches the CE
term's."  *** MEASURED, NOT ASSUMED.  See calibrate_lambda(): the student IS the frozen
anchor at step 0 (zero new parameters), so whether this is even well-defined depends on
how much of the step-0 student/anchor discrepancy is real.  The two gradient norms are
logged; a degenerate ratio is reported, never silently replaced. ***

CLASS EXCLUSION.  terrain and manmade are dropped from the distillation loss in arm B.
INPUT.  no crop; the full 360 deg sweep, matching inference.
RARE-CLASS SAMPLING.  frame-level, by teacher-predicted presence; identical for B, C, D.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/data/wuyou/livo_sem/src")
import pointcept_ext as PX                                      # noqa: E402
from pointcept.models.builder import MODELS, build_model        # noqa: E402
from pointcept.models.utils.structure import Point              # noqa: E402
from pointcept.models.losses.lovasz import _lovasz_softmax      # noqa: E402
from pointcept.datasets.builder import DATASETS                 # noqa: E402
from filter_e import filter_E_mask, F_INFRUSTUM, F_VISIBLE, \
    F_DEPTHEDGE, F_RANGE_LT50                                   # noqa: E402
from random_supervise import random_supervise_mask               # noqa: E402

WEIGHT_DIR = "/data/wuyou/livo_sem/weights/nuscenes-semseg-pt-v3m1-0-base"
CKPT_PATH = WEIGHT_DIR + "/model/model_best.pth"

COMMON9_NAMES = PX.COMMON9_NAMES                 # 9 names, label_spaces order
GROUPS = PX.COMMON9_TO_NUSC16                    # list[9] of nuScenes-16 index lists
NUSC16 = 16


def _load_released_state_dict():
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def group_matrix(device=None, dtype=torch.float32):
    """(16, 9) 0/1 matrix G with G[k, c] = 1 iff nuScenes class k is in common-9 group c.
    Column 11 (other_flat) is all-zero on purpose."""
    G = torch.zeros(NUSC16, len(GROUPS), dtype=dtype)
    for c, ks in enumerate(GROUPS):
        for k in ks:
            G[k, c] = 1.0
    return G.to(device) if device is not None else G


# ===================================================================== #
#  MODEL
# ===================================================================== #
@MODELS.register_module()
class DistilSegmentorMiB(nn.Module):
    def __init__(self,
                 backbone,
                 backbone_out_channels=64,
                 weight=CKPT_PATH,
                 ignore_index=-1,
                 exclude_classes=(),        # common-9 names dropped from the sup. loss
                 use_lovasz=True,
                 kl_lambda=None,            # None -> calibrate at step 0 and log it
                 kl_enabled=True,
                 freeze_backbone=False,
                 eps=1e-8):
        super().__init__()
        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(backbone_out_channels, NUSC16)
        # THE ANCHOR IS THE PROJECT'S VERIFIED FROZEN PTv3, not a second student:
        # shuffle_orders pinned False and fp16 weights, exactly as
        # src/ptv3_loader_verified.build_ptv3(shuffle_orders=False, half=True) builds
        # the model that produced arm A.  Two reasons, both measured, not stylistic:
        #  * v1.5.1 applies shuffle_orders inside forward() ungated by self.training, so
        #    an anchor with it True would draw a FRESH serialisation order every step and
        #    the KL would be measuring that noise rather than the student's drift.
        #  * spconv 2.3.8 in EVAL mode cannot tune implicit_gemm with fp16 activations
        #    against fp32 weights ("can't find suitable algorithm"); the verified loader
        #    solves it by casting the whole model to fp16, so the anchor does the same
        #    and runs OUTSIDE autocast.
        anchor_bb = dict(backbone)
        anchor_bb["shuffle_orders"] = False
        self.frozen_backbone = build_model(anchor_bb)
        self.frozen_head = nn.Linear(backbone_out_channels, NUSC16)

        sd = _load_released_state_dict()
        bk = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        hd = {k[len("seg_head."):]: v for k, v in sd.items() if k.startswith("seg_head.")}
        assert len(bk) + len(hd) == len(sd), ("unexpected keys in checkpoint",
                                              sorted(set(sd) - set(
                                                  ["backbone." + k for k in bk] +
                                                  ["seg_head." + k for k in hd]))[:8])
        for mod, s, nm in ((self.backbone, bk, "backbone"),
                           (self.seg_head, hd, "seg_head"),
                           (self.frozen_backbone, bk, "frozen_backbone"),
                           (self.frozen_head, hd, "frozen_head")):
            info = mod.load_state_dict(s, strict=True)     # MUST stay strict (C1)
            del info
        self._n_loaded = len(sd)
        print("[DistilSegmentorMiB] loaded %d released tensors into student AND anchor "
              "(strict=True on all four loads)" % len(sd), flush=True)

        for p in self.frozen_backbone.parameters():
            p.requires_grad_(False)
        for p in self.frozen_head.parameters():
            p.requires_grad_(False)
        self.frozen_backbone.eval().half()
        self.frozen_head.eval().half()

        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            # Arm B1: the trunk is frozen, so the student backbone IS the verified
            # frozen PTv3 -- fp16 weights, eval mode, run outside autocast, exactly like
            # the anchor.  Keeping it fp32-in-eval-under-autocast is what spconv 2.3.8
            # cannot tune; keeping it in TRAIN mode would leave drop-path 0.3 active in a
            # trunk that is supposed to be frozen.  Only seg_head (1040 params) trains.
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval().half()

        self.register_buffer("G", group_matrix(), persistent=False)
        keep = torch.ones(len(GROUPS), dtype=torch.bool)
        for nme in exclude_classes:
            keep[COMMON9_NAMES.index(nme)] = False
        self.register_buffer("keep_class", keep, persistent=False)
        self.exclude_classes = list(exclude_classes)
        self.ignore_index = ignore_index
        self.use_lovasz = bool(use_lovasz)
        self.kl_enabled = bool(kl_enabled)
        self.kl_lambda = kl_lambda
        self.eps = float(eps)
        self._cal_log = None
        if self.kl_lambda is not None:
            # PINNED, NOT CALIBRATED.  Only arm R arrives here.  The step-0 calibration
            # is a MEASUREMENT of a ratio of gradient norms on one batch, and
            # out/v04/lambda_calibration.jsonl shows it spanning 0.441 / 0.969 / 1.531
            # across three runs of the SAME full-fine-tune configuration -- a 3.5x range
            # on a knob that sets anti-forgetting strength, against a decision band of
            # +-0.60 mIoU.  Arm R's only meaningful comparison is against arm D, so it
            # takes arm D's calibrated value verbatim; letting it self-calibrate would
            # make R differ from D in TWO respects (supervision geometry AND anchor
            # strength) and a two-factor contrast attributes nothing.
            print("[lambda-PINNED] kl_lambda = %.17g taken from the config; the step-0 "
                  "calibration is SKIPPED for this arm" % float(self.kl_lambda),
                  flush=True)
            with open("/data/wuyou/livo_sem/out/v04/lambda_calibration.jsonl", "a") as fh:
                fh.write(json.dumps(dict(pinned=True,
                                         kl_lambda=float(self.kl_lambda),
                                         source="arm D, out/v04/lambda_calibration.jsonl",
                                         grad_norm_sup=None, grad_norm_kl=None,
                                         ratio=None, kl_value=None,
                                         exclude=self.exclude_classes,
                                         freeze_backbone=self.freeze_backbone)) + "\n")
        # RARE-CLASS SELF-CHECK: points per common-9 class that actually entered the
        # supervised loss this epoch.  The frozen design requires this to be logged.
        self.register_buffer("cls_count", torch.zeros(len(GROUPS), dtype=torch.float64),
                             persistent=False)
        self.register_buffer("kl_count", torch.zeros(1, dtype=torch.float64),
                             persistent=False)

    # ----------------------------------------------------------------- #
    def train(self, mode=True):
        super().train(mode)
        self.frozen_backbone.eval()       # the anchor is NEVER in train mode
        self.frozen_head.eval()
        if self.freeze_backbone:
            self.backbone.eval()          # no drop-path / no BN updates in a frozen trunk
        return self

    def _logits16(self, input_dict):
        if self.freeze_backbone:
            d = dict(input_dict)
            d["coord"] = d["coord"].half()
            d["feat"] = d["feat"].half()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                point = self.backbone(Point(d))
                feat = point.feat.float()
            return self.seg_head(feat)
        point = self.backbone(Point(dict(input_dict)))
        return self.seg_head(point.feat)

    @torch.no_grad()
    def _anchor_logits16(self, input_dict):
        keys = ("coord", "grid_coord", "feat", "offset")
        sub = {k: input_dict[k] for k in keys if k in input_dict}
        sub["coord"] = sub["coord"].half()
        sub["feat"] = sub["feat"].half()
        with torch.cuda.amp.autocast(enabled=False):
            point = self.frozen_backbone(Point(sub))
            return self.frozen_head(point.feat).float()

    def _p9(self, logits16):
        # softmax is on autocast's fp32 list and the input is forced .float(), so p16 is
        # fp32.  The marginalisation is done with slice-sums rather than `p16 @ G`
        # because a matmul inside autocast would be re-cast to fp16 and silently halve
        # the precision of the entire supervised loss.  Exact, and 9 tiny reductions.
        p16 = torch.softmax(logits16.float(), dim=-1)
        p9 = torch.stack([p16[:, ks].sum(dim=1) for ks in GROUPS], dim=1)
        return p9, p16                      # sum_c p9(c) < 1 by design (other_flat)

    def _sup_loss(self, p9, segment):
        """MiB-style NLL on the marginalised 9-way probability, + Lovasz on the same."""
        m = segment >= 0
        if self.keep_class.numel() and not bool(self.keep_class.all()):
            s = segment.clamp_min(0)
            m = m & self.keep_class[s]
        n = int(m.sum())
        if n == 0:
            z = p9.sum() * 0.0
            return z, z, 0
        q = p9[m]
        t = segment[m].long()
        if self.training:                  # validation must not pollute the epoch audit
            with torch.no_grad():
                self.cls_count += torch.bincount(t, minlength=len(GROUPS)).double()
        ce = -(torch.log(q.gather(1, t[:, None]).squeeze(1) + self.eps)).mean()
        lz = q.sum() * 0.0
        if self.use_lovasz:
            lz = _lovasz_softmax(q, t, classes="present", per_image=False)
        return ce, lz, n

    def _kl(self, logits_s16, logits_t16, frustum):
        """KL(anchor || student) on the FULL 16-dim simplex, out-of-frustum points."""
        m = ~frustum
        n = int(m.sum())
        if n == 0 or not self.kl_enabled:
            return logits_s16.sum() * 0.0, 0
        ls = torch.log_softmax(logits_s16[m].float(), dim=-1)
        lt = torch.log_softmax(logits_t16[m].float(), dim=-1)
        pt = lt.exp()
        return (pt * (lt - ls)).sum(-1).mean(), n

    # ----------------------------------------------------------------- #
    def forward(self, input_dict):
        logits16 = self._logits16(input_dict)
        p9, _p16 = self._p9(logits16)

        if not self.training and "segment" not in input_dict:
            return dict(seg_logits=torch.log(p9 + self.eps))

        segment = input_dict["segment"]
        ce, lz, n_sup = self._sup_loss(p9, segment)
        sup = ce + lz

        kl = logits16.sum() * 0.0
        n_kl = 0
        if self.kl_enabled and "frustum" in input_dict:
            fr = input_dict["frustum"].bool()
            anchor = self._anchor_logits16(input_dict)
            kl, n_kl = self._kl(logits16, anchor, fr)
            with torch.no_grad():
                self.kl_count += float(n_kl)

        if (self.training and self.kl_enabled and self.kl_lambda is None
                and n_sup > 0 and n_kl > 0):
            # A first batch with no supervised point would calibrate lambda to 0 and
            # silently disable anti-forgetting for the whole run.  Wait for a real one.
            self.kl_lambda = self._calibrate(sup, kl)

        lam = 0.0 if self.kl_lambda is None else float(self.kl_lambda)
        loss = sup + lam * kl

        if self.training:
            # every value must be a tensor: InformationWriter calls .item() on all of
            # them (hooks/misc.py:111), so a plain int crashes the logger.
            dev = loss.device
            return dict(loss=loss, ce=ce.detach(), lovasz=lz.detach(),
                        kl=kl.detach(),
                        n_sup=torch.tensor(float(n_sup), device=dev),
                        n_kl=torch.tensor(float(n_kl), device=dev))
        return dict(loss=loss, seg_logits=torch.log(p9 + self.eps))

    # ----------------------------------------------------------------- #
    def _calibrate(self, sup, kl):
        """lambda such that ||d(lambda*KL)/dtheta|| == ||d(sup)/dtheta|| at step 0.

        MEASURED and LOGGED.  If the anchor and the student are bit-identical at step 0
        -- which they are by construction whenever nothing stochastic separates them --
        ||dKL|| is exactly 0 and the ratio is NOT DEFINED.  That case is reported, not
        papered over: kl_lambda stays None and the caller must decide.
        """
        ps = [p for p in self.parameters() if p.requires_grad]
        gs = torch.autograd.grad(sup, ps, retain_graph=True, allow_unused=True)
        ns = torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)).item()
        try:
            gk = torch.autograd.grad(kl, ps, retain_graph=True, allow_unused=True)
            nk = torch.sqrt(sum((g.float() ** 2).sum() for g in gk if g is not None)).item()
        except RuntimeError:
            nk = 0.0
        self._cal_log = dict(grad_norm_sup=ns, grad_norm_kl=nk,
                             ratio=(ns / nk if nk > 0 else None),
                             kl_value=float(kl.detach()))
        print("[lambda-calibration] ||grad sup|| = %.6g   ||grad KL|| = %.6g   "
              "KL = %.6g   lambda = %s" %
              (ns, nk, float(kl.detach()),
               ("%.6g" % (ns / nk)) if nk > 0 else "UNDEFINED (||grad KL|| == 0)"),
              flush=True)
        with open("/data/wuyou/livo_sem/out/v04/lambda_calibration.jsonl", "a") as fh:
            fh.write(json.dumps(dict(self._cal_log,
                                     exclude=self.exclude_classes,
                                     freeze_backbone=self.freeze_backbone)) + "\n")
        if nk <= 0:
            return None
        return ns / nk


# ===================================================================== #
#  DATASET
# ===================================================================== #
@DATASETS.register_module()
class DistilSemanticKITTIDataset(PX.SemanticKITTICommon9Dataset):
    """SemanticKITTI in COMMON-9 with the v0.4 supervision sources.

    label_source:
      "gt"      arms C/D     -- SemanticKITTI GT through the common-9 LUT
      "pseudo"  arms B0/B1   -- the 2D teacher, cleaned by filter E
    supervise:
      "all"       arm C      -- every point carries loss
      "frustum"   arm D      -- only camera-frustum points carry loss
      "filterE"   arm B      -- only the points filter E keeps
      "random"    arm R      -- a per-frame RANDOM subset of the same SIZE as arm D's
                               frustum, over the full 360 deg sweep (see
                               src/random_supervise.py)

    `frustum` (bool per point) is ALWAYS emitted: the anti-forgetting KL is defined on
    its complement and must survive GridSample together with coord/strength/segment.

    RARE-CLASS SAMPLING is frame-level and label-source-independent: the multiplier is
    read from a JSON built once from the TEACHER's predicted presence, so arms B, C and
    D see bit-identical frame orders.  UNDER-SPECIFIED CONSTANT, DECLARED: the frozen
    design fixes the mechanism ("frame-level resampling by teacher-predicted presence of
    person / two_wheeler / large_vehicle") but not the multiplier.  `sample_weights_json`
    carries it explicitly so the number is visible in the config rather than invented here.
    """

    def __init__(self, pseudo_root=None, filter_spec=None, label_source="gt",
                 supervise="all", sample_weights_json=None, **kw):
        self.pseudo_root = pseudo_root
        self.filter_spec = dict(filter_spec or {})
        self.label_source = label_source
        self.supervise = supervise
        self.sample_weights_json = sample_weights_json
        assert label_source in ("gt", "pseudo"), label_source
        assert supervise in ("all", "frustum", "filterE", "random"), supervise
        super().__init__(**kw)

    RESAMPLE_SEED = 20260922      # fixed, so arms B, C and D get BIT-IDENTICAL orders

    def get_data_list(self):
        base = super().get_data_list()
        if not self.sample_weights_json or self.split != "train":
            return base
        w = json.load(open(self.sample_weights_json))["weights"]
        ww = np.array([float(w.get("%s/%s" % self._seq_frame(p), 1.0)) for p in base])
        # LENGTH-PRESERVING resampling: draw len(base) frames with replacement with
        # probability proportional to the rare-class weight.  Duplicating instead would
        # multiply the epoch (and the 7.6 h/arm budget) by the mean weight; the design
        # says "frame-level resampling", not "a longer epoch".  Fixed seed, so the
        # identical sampling really is identical across arms.
        rng = np.random.default_rng(self.RESAMPLE_SEED)
        idx = rng.choice(len(base), size=len(base), replace=True, p=ww / ww.sum())
        out = [base[i] for i in idx]
        uniq = len(set(idx.tolist()))
        print("[DistilSemanticKITTIDataset] rare-class resampling: %d frames drawn from "
              "%d (%d distinct, mean weight %.3f, seed %d)"
              % (len(out), len(base), uniq, ww.mean(), self.RESAMPLE_SEED), flush=True)
        return out

    def _pseudo(self, seq, frame):
        z = np.load(os.path.join(self.pseudo_root, seq, "f%06d.npz" % int(frame)))
        return z["t9"], z["m9"], z["conf"].astype(np.float32), z["flags"]

    def get_data(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = (scan[:, 3] * self.strength_scale).reshape(-1, 1)
        seq, frame = self._seq_frame(path)
        n = coord.shape[0]

        t9, m9, conf, flags = self._pseudo(seq, frame)
        assert t9.shape[0] == n, (seq, frame, n, t9.shape)
        frustum = (flags & F_INFRUSTUM) > 0

        if self.label_source == "gt":
            lf = os.path.join(os.path.dirname(os.path.dirname(path)),
                              "labels", frame + ".label")
            raw = np.fromfile(lf, dtype=np.uint32) & 0xFFFF
            segment = self.raw_lut[raw].astype(np.int32)
        else:
            segment = t9.astype(np.int32)
            segment[segment < 0] = self.ignore_index

        if self.supervise == "frustum":
            segment = np.where(frustum, segment, self.ignore_index).astype(np.int32)
        elif self.supervise == "filterE":
            keep = filter_E_mask(t9, m9, conf, flags, self.filter_spec)
            segment = np.where(keep, segment, self.ignore_index).astype(np.int32)
        elif self.supervise == "random":
            # ARM R.  Same NUMBER of supervised points as arm D in THIS frame, drawn
            # uniformly at random over the whole 360 deg sweep instead of taken from
            # the camera frustum.  The count comes from arm D's own mask, recomputed
            # two lines up, so the arms cannot drift; the draw is seeded from
            # (sequence, frame) alone, so the subset is the SAME in all 10 epochs.
            sel, _k = random_supervise_mask(frustum, segment >= 0, seq, frame)
            segment = np.where(sel, segment, self.ignore_index).astype(np.int32)
            # The KL anchor is defined on the complement of the SELECTION mask, which
            # is what the model reads out of the `frustum` key -- i.e. on every
            # NON-supervised point, the same rule arm D obeys, applied to a scattered
            # set instead of a contiguous one.  NOTE the deliberate overload: in this
            # arm `frustum` no longer carries the camera frustum.  It is the only way
            # to express "KL on the unsupervised points" without touching the model.
            frustum = sel

        return dict(coord=coord, strength=strength, segment=segment,
                    frustum=frustum.astype(np.int32).reshape(-1))


# ===================================================================== #
#  HOOK -- the per-epoch rare-class self-check the frozen design requires
# ===================================================================== #
from pointcept.engines.hooks.builder import HOOKS              # noqa: E402
from pointcept.engines.hooks.default import HookBase           # noqa: E402


@HOOKS.register_module()
class RareClassAudit(HookBase):
    """Self-check each epoch: the number of person / two_wheeler / large_vehicle points
    that ACTUALLY entered the loss.  Counts are accumulated inside the model and dumped
    here, per epoch, to <save_path>/rare_class_audit.jsonl.

    NO THRESHOLD IS APPLIED.  The frozen design says "if below threshold, void the
    epoch" but does not state the threshold; inventing one would be substituting a
    choice.  The counts are recorded so that decision can be made on evidence.
    """

    def __init__(self, out_name="rare_class_audit.jsonl"):
        self.out_name = out_name

    def _model(self):
        m = self.trainer.model
        return m.module if hasattr(m, "module") else m

    def before_epoch(self):
        m = self._model()
        if hasattr(m, "cls_count"):
            m.cls_count.zero_()
            m.kl_count.zero_()

    def after_epoch(self):
        m = self._model()
        if not hasattr(m, "cls_count"):
            return
        c = m.cls_count.detach().cpu().numpy().astype("int64")
        rec = dict(epoch=int(self.trainer.epoch) + 1,
                   kl_points=int(m.kl_count.item()),
                   supervised_points=int(c.sum()),
                   per_class={n: int(v) for n, v in zip(COMMON9_NAMES, c)},
                   kl_lambda=(None if m.kl_lambda is None else float(m.kl_lambda)))
        pth = os.path.join(self.trainer.cfg.save_path, self.out_name)
        with open(pth, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.trainer.logger.info(
            "[RareClassAudit] epoch %d  sup=%d  kl=%d  person=%d two_wheeler=%d "
            "large_vehicle=%d" % (rec["epoch"], rec["supervised_points"],
                                  rec["kl_points"], rec["per_class"]["person"],
                                  rec["per_class"]["two_wheeler"],
                                  rec["per_class"]["large_vehicle"]))
