#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_distil.py -- build the v0.4 model and dataset EXACTLY as the config does, run one
training step, and MEASURE the two things that cannot be assumed:

  1. does the 16-way head + MiB marginalisation load strict and produce a sane p9?
  2. what is the lambda calibration actually worth at step 0?  The student IS the frozen
     anchor at step 0 (zero new parameters), so ||grad KL|| may be exactly 0, in which
     case the frozen design's "match the CE gradient norm at step 0" is NOT DEFINED.
     That is reported here as a number, never guessed.

Runs on seq 04 alone (271 frames, already on disk), so it needs none of the download.
"""
import os, sys, json, argparse
sys.path.insert(0, "/data/wuyou/livo_sem/src")
import numpy as np
import torch

import pointcept_ext          # noqa: F401
import distil_ext             # noqa: F401
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.datasets.utils import point_collate_fn

CFG = ("/data/wuyou/livo_sem/src/Pointcept_v151/configs/semantic_kitti/"
       "semseg-pt-v3m1-distil-common9.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B0",
                    choices=["B0", "B1", "C", "D"])
    ap.add_argument("--shuffle-orders", default="", help="'true'/'false' to override")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()

    cfg = Config.fromfile(CFG)
    if a.shuffle_orders:
        cfg.model.backbone.shuffle_orders = (a.shuffle_orders.lower() == "true")
    if a.arm == "B1":
        cfg.model.freeze_backbone = True
        cfg.model.exclude_classes = ("terrain", "manmade")
    elif a.arm == "B0":
        cfg.model.exclude_classes = ("terrain", "manmade")
    elif a.arm == "D":
        cfg.data.train.label_source = "gt"; cfg.data.train.supervise = "frustum"
    elif a.arm == "C":
        cfg.data.train.label_source = "gt"; cfg.data.train.supervise = "all"
    # seq 04 only, and no resampling json yet
    cfg.data.train.sample_weights_json = None
    import pointcept_ext as PX
    PX.SemanticKITTICommon9Dataset.SPLIT2SEQ = dict(train=[4], val=[8], test=[7])

    print("shuffle_orders =", cfg.model.backbone.shuffle_orders,
          " arm =", a.arm, " exclude =", cfg.model.exclude_classes, flush=True)
    ds = build_dataset(cfg.data.train)
    print("train frames:", len(ds), flush=True)
    model = build_model(cfg.model).cuda()
    model.train()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print("params trainable %.2f M / total %.2f M" % (n_tr / 1e6, n_all / 1e6), flush=True)

    batch = point_collate_fn([ds[i] for i in range(cfg.batch_size)], mix_prob=0.0)
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda(non_blocking=True)
    print("batch keys:", {k: tuple(v.shape) for k, v in batch.items()
                          if isinstance(v, torch.Tensor)}, flush=True)
    seg = batch["segment"]; fr = batch["frustum"].bool()
    print("points %d  supervised %d (%.2f %%)  in-frustum %d (%.2f %%)  "
          "out-of-frustum (KL) %d" %
          (seg.numel(), int((seg >= 0).sum()), 100.0 * float((seg >= 0).sum()) / seg.numel(),
           int(fr.sum()), 100.0 * float(fr.sum()) / seg.numel(), int((~fr).sum())), flush=True)

    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    for step in range(a.steps):
        with torch.cuda.amp.autocast(enabled=True):
            out = model(batch)
        print("step %d  loss %.5f  ce %.5f  lovasz %.5f  kl %.6g  n_sup %d  n_kl %d  "
              "lambda %s" %
              (step, float(out["loss"]), float(out["ce"]), float(out["lovasz"]),
               float(out["kl"]), out["n_sup"], out["n_kl"],
               model.kl_lambda), flush=True)
        opt.zero_grad()
        scaler.scale(out["loss"]).backward()
        scaler.step(opt); scaler.update()
    print("peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)

    # Pointcept's SemSegEvaluator runs the model in eval() under torch.no_grad() and
    # WITHOUT autocast (hooks/evaluator.py:118), i.e. the pure-fp32 path.  Wrapping eval
    # in autocast is what spconv 2.3.8 cannot tune (fp16 activations, fp32 weights), so
    # the check below reproduces the evaluator, not a variant of it.
    model.eval()
    with torch.no_grad():
        o = model(batch)
    pred = o["seg_logits"].max(1)[1]
    seg = batch["segment"]
    m = seg >= 0
    print("eval seg_logits:", tuple(o["seg_logits"].shape),
          " point-acc on supervised pts %.4f" % float((pred[m] == seg[m]).float().mean()),
          flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
