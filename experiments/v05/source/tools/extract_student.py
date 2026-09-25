#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/extract_student.py -- v0.5.  Put a trained DistilSegmentorMiB checkpoint back into
the DEPLOYED plain segmentor (DefaultSegmentorV2, built by ptv3_loader_verified.build_ptv3
on Pointcept v1.5.1) and PROVE it is the same model before it goes near the pipeline.

Run under the ptv3 conda python:
    /data/miniconda3/envs/ptv3/bin/python tools/extract_student.py \
        --ckpt exp/sk/armB0/model/model_best.pth --tag B0 \
        --out weights/v05/B0_student.pth --report out/v05/extract_B0.json \
        [--neg weights/v05/<other>_student.pth]

WHAT A TRAINED CHECKPOINT CONTAINS (measured on exp/sk/armB0/model/model_best.pth)
    976 tensors, no `module.` prefix:
      backbone.*         486   fp32 (473) + int64 BatchNorm counters (13)   student trunk
      seg_head.*           2   fp32                                         student 16-way head
      frozen_backbone.*  486   fp16                                         anti-forgetting ANCHOR
      frozen_head.*        2   fp16                                         anchor head
    plus optimizer / scheduler / scaler state (that is the 647 MB against 529 MB).
The anchor is the released model, kept only for the training-time KL.  It must NOT be
deployed.  The student's backbone.* + seg_head.* are exactly the 488 tensors the released
checkpoint carries: DistilSegmentorMiB adds ZERO parameters, the MiB marginalisation
p9(c) = sum_{k in g(c)} p16(k) lives in the LOSS only, and the deployed decision rule
(argmax over the 16-way head, confidence = max softmax) is the one tools/cache_trained.py
scored offline (`model._logits16`).

THE FOUR CHECKS -- all must pass or the tool exits non-zero
  S1 structure  student key set == released key set, same shapes, same dtypes; strict=True
                load into DefaultSegmentorV2: 0 missing / 0 unexpected, 488/488.
  S2 weights    every one of the 488 tensors of the loaded plain model is torch.equal to
                the corresponding tensor of the ORIGINAL DistilSegmentorMiB loaded exactly
                the way tools/cache_trained.py loads it (the object the offline mIoU came
                from), compared in fp32 and again after the deployed .half().
  S3 code path  repr(plain.backbone) == repr(distil.backbone), repr of the heads equal,
                identical (name, shape, dtype) lists for parameters and buffers -- so
                DefaultSegmentorV2.forward and DistilSegmentorMiB._logits16 are the same
                function of the same tensors.
  S4 behaviour  on real seq07 scans, on the SAME voxelised input tensors: argmax agreement
                plain-vs-distil is compared with the SAME-MODEL repeat agreement (the forward
                is non-deterministic, CRITICAL_CONSTRAINTS R2: an unstable CUDA sort inside
                SerializedPooling), stratified by logit margin, plus |delta logit| stats.
                Equivalence means the cross-model numbers sit inside the within-model ones
                (in particular ~100 % agreement wherever the margin is large).  A NEGATIVE
                CONTROL (a different student, --neg) shows what a genuinely different model
                looks like on the same test.  Finally the DEPLOYED path itself --
                ptv3_worker.Segmenter with PTV3_CKPT, GPU voxeliser, fast Hilbert, fp16 --
                is run on the same scans and scored against SemanticKITTI GT (common-9) next
                to the original DistilSegmentorMiB.
"""
import os, sys, json, time, hashlib, argparse
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np

R = "/data/livo_sem"
RELEASED = R + "/weights/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth"
DISTIL_CFG = (R + "/src/Pointcept_v151/configs/semantic_kitti/"
              "semseg-pt-v3m1-distil-common9.py")
SCANS = R + "/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/%010d.bin"
LABELS = R + "/data/odometry/dataset/sequences/07/labels/%06d.label"
INTENSITY_SCALE = 0.2      # CRITICAL_CONSTRAINTS C3
GRID = 0.05
STUDENT_PREFIXES = ("backbone.", "seg_head.")
ANCHOR_PREFIXES = ("frozen_backbone.", "frozen_head.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_module(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def voxelize(coord, strength, grid):
    """Verbatim tools/cache_trained.py: one representative (first in order) per cell."""
    g = np.floor(coord / grid).astype(np.int64)
    g -= g.min(0)
    key = (g[:, 0] * (g[:, 1].max() + 1) + g[:, 1]) * (g[:, 2].max() + 1) + g[:, 2]
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return (np.ascontiguousarray(coord[first]), np.ascontiguousarray(strength[first]),
            np.ascontiguousarray(g[first]), inv)


def read_scan(f):
    return np.fromfile(SCANS % f, dtype=np.float32).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--neg", default=None, help="another extracted student = negative control")
    ap.add_argument("--frames", default="0:1101:11",
                    help="comma list, or start:stop:step (default every 11th frame, 101 frames)")
    a = ap.parse_args()
    if ":" in a.frames:
        f0, f1, st = (int(x) for x in a.frames.split(":"))
        frames = list(range(f0, f1, st))
    else:
        frames = [int(x) for x in a.frames.split(",")]
    rep = dict(tag=a.tag, ckpt=a.ckpt, out=a.out, frames=frames,
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    fails = []

    import torch
    import pointcept_ext                      # noqa: F401  stubs + sys.path + yapf shim
    import distil_ext                         # noqa: F401  registers DistilSegmentorMiB
    from pointcept.utils.config import Config
    from pointcept.models import build_model
    from ptv3_loader_verified import build_ptv3, CONFIG_PATH, NUSCENES_CLASSES
    import label_spaces as LS

    # ------------------------------------------------------------------ partition
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd_all = strip_module(ck["state_dict"] if "state_dict" in ck else ck)
    student = {k: v for k, v in sd_all.items() if k.startswith(STUDENT_PREFIXES)}
    anchor = {k: v for k, v in sd_all.items() if k.startswith(ANCHOR_PREFIXES)}
    other = sorted(set(sd_all) - set(student) - set(anchor))
    rep["source"] = dict(sha256=sha256(a.ckpt), bytes=os.path.getsize(a.ckpt),
                         top_level_keys=sorted(ck.keys()) if isinstance(ck, dict) else None,
                         epoch=(int(ck["epoch"]) if isinstance(ck, dict) and "epoch" in ck else None),
                         best_metric_value=(float(ck["best_metric_value"])
                                            if isinstance(ck, dict) and "best_metric_value" in ck else None),
                         n_tensors=len(sd_all), n_student=len(student), n_anchor=len(anchor),
                         n_other=len(other), other_keys=other[:10],
                         student_dtypes={str(d): int(sum(1 for v in student.values() if v.dtype == d))
                                         for d in set(v.dtype for v in student.values())},
                         anchor_dtypes={str(d): int(sum(1 for v in anchor.values() if v.dtype == d))
                                        for d in set(v.dtype for v in anchor.values())},
                         student_bytes=int(sum(v.numel() * v.element_size() for v in student.values())),
                         anchor_bytes=int(sum(v.numel() * v.element_size() for v in anchor.values())))
    print("[partition] %d tensors: student %d, anchor %d, other %d  (epoch %s, best %s)"
          % (len(sd_all), len(student), len(anchor), len(other),
             rep["source"]["epoch"], rep["source"]["best_metric_value"]))
    if other:
        fails.append("unexpected non-student/non-anchor keys: %s" % other[:5])

    # ------------------------------------------------------------------ S1 structure
    rel = strip_module(torch.load(RELEASED, map_location="cpu", weights_only=False)["state_dict"])
    same_keys = set(student) == set(rel)
    shape_ok = same_keys and all(tuple(student[k].shape) == tuple(rel[k].shape) for k in rel)
    dtype_ok = same_keys and all(student[k].dtype == rel[k].dtype for k in rel)
    rep["S1"] = dict(n_student=len(student), n_released=len(rel), same_key_set=bool(same_keys),
                     same_shapes=bool(shape_ok), same_dtypes=bool(dtype_ok),
                     missing_vs_released=sorted(set(rel) - set(student))[:10],
                     extra_vs_released=sorted(set(student) - set(rel))[:10])
    if not (same_keys and shape_ok and dtype_ok):
        fails.append("S1 key/shape/dtype set differs from the released checkpoint")
    print("[S1] key set == released: %s  shapes: %s  dtypes: %s  (%d / %d)"
          % (same_keys, shape_ok, dtype_ok, len(student), len(rel)))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    reuse = False
    if os.path.exists(a.out):
        prev = torch.load(a.out, map_location="cpu", weights_only=False)["state_dict"]
        reuse = set(prev) == set(student) and all(torch.equal(prev[k], student[k]) for k in student)
        print("[write] %s exists; tensors identical to a fresh extraction: %s" % (a.out, reuse))
        if not reuse:
            os.rename(a.out, a.out + ".stale")
    rep["reused_existing_file"] = reuse
    if not reuse:
      torch.save(dict(state_dict=student,
                    meta=dict(source=a.ckpt, source_sha256=rep["source"]["sha256"],
                              epoch=rep["source"]["epoch"],
                              best_metric_value=rep["source"]["best_metric_value"],
                              extracted=rep["time"], tag=a.tag,
                              rule="student backbone.* + seg_head.* only; frozen_* dropped")),
               a.out)
    # (torch.save is only reached when the file was absent or stale)
    rep["out_sha256"] = sha256(a.out)
    rep["out_bytes"] = os.path.getsize(a.out)
    print("[write] %s  %d bytes  sha256 %s" % (a.out, rep["out_bytes"], rep["out_sha256"][:16]))

    # explicit strict load into a fresh DefaultSegmentorV2 (fp32, CPU), with the report
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.backbone.shuffle_orders = False
    m_cpu = build_model(cfg.model)
    sd_out = torch.load(a.out, map_location="cpu", weights_only=False)["state_dict"]
    info = m_cpu.load_state_dict(sd_out, strict=True)
    n_model = len(m_cpu.state_dict())
    rep["S1"].update(strict_load_ok=True, missing=list(info.missing_keys),
                     unexpected=list(info.unexpected_keys),
                     model_tensors=n_model, ckpt_tensors=len(sd_out))
    print("[S1] strict=True load into DefaultSegmentorV2: %d/%d tensors, %d missing, %d unexpected"
          % (len(sd_out), n_model, len(info.missing_keys), len(info.unexpected_keys)))
    if n_model != len(sd_out) or info.missing_keys or info.unexpected_keys:
        fails.append("S1 strict load mismatch")

    # ------------------------------------------------------------------ the two models
    dev = "cuda"
    plain, cfg_p = build_ptv3(device=dev, shuffle_orders=False, half=True, ckpt_path=a.out)
    assert getattr(plain, "_ckpt_path", None) == a.out, "build_ptv3 did not take ckpt_path"
    rep["plain_loaded_from"] = plain._ckpt_path
    rep["plain_n_params"] = int(sum(p.numel() for p in plain.parameters()))

    # ORIGINAL DistilSegmentorMiB, loaded exactly as tools/cache_trained.py does
    cfgd = Config.fromfile(DISTIL_CFG)
    cfgd.model.backbone.shuffle_orders = False
    cfgd.model.kl_enabled = False
    distil = build_model(cfgd.model)          # __init__ loads the RELEASED weights (strict)
    sd_nofrozen = {k: v for k, v in sd_all.items() if not k.startswith("frozen_")}
    info2 = distil.load_state_dict(sd_nofrozen, strict=False)
    missing2 = [k for k in info2.missing_keys if not k.startswith("frozen_")]
    assert not missing2, ("missing non-frozen keys", missing2[:10])
    assert not info2.unexpected_keys, ("unexpected keys", info2.unexpected_keys[:10])
    rep["distil_load"] = dict(loaded=len(sd_nofrozen), frozen_left_at_init=len(info2.missing_keys))

    # ------------------------------------------------------------------ S2 weights (fp32)
    ds32 = distil.state_dict()
    sd_plain32 = m_cpu.state_dict()
    neq32 = [k for k in sd_plain32 if not torch.equal(sd_plain32[k], ds32[k])]
    print("[S2] fp32: %d/%d tensors torch.equal to the DistilSegmentorMiB student (%d differ)"
          % (len(sd_plain32) - len(neq32), len(sd_plain32), len(neq32)))
    distil = distil.to(dev).eval().half()
    for p in distil.parameters():
        p.requires_grad_(False)
    ds16 = distil.state_dict()
    ps16 = plain.state_dict()
    neq16 = [k for k in ps16 if not torch.equal(ps16[k], ds16[k])]
    print("[S2] fp16 (deployed dtype): %d/%d equal (%d differ)"
          % (len(ps16) - len(neq16), len(ps16), len(neq16)))
    rep["S2"] = dict(n=len(ps16), n_equal_fp32=len(sd_plain32) - len(neq32),
                     n_equal_fp16=len(ps16) - len(neq16), differ_fp32=neq32[:10], differ_fp16=neq16[:10])
    if neq32 or neq16:
        fails.append("S2 tensor mismatch")

    # ------------------------------------------------------------------ S3 code path
    def sig(mod):
        return ([(n, tuple(p.shape), str(p.dtype)) for n, p in mod.named_parameters()],
                [(n, tuple(b.shape), str(b.dtype)) for n, b in mod.named_buffers()])
    r_bb = repr(plain.backbone) == repr(distil.backbone)
    r_hd = repr(plain.seg_head) == repr(distil.seg_head)
    s_bb = sig(plain.backbone) == sig(distil.backbone)
    s_hd = sig(plain.seg_head) == sig(distil.seg_head)
    rep["S3"] = dict(backbone_repr_equal=bool(r_bb), head_repr_equal=bool(r_hd),
                     backbone_named_tensors_equal=bool(s_bb), head_named_tensors_equal=bool(s_hd),
                     plain_type=type(plain).__name__, distil_type=type(distil).__name__,
                     backbone_type=type(plain.backbone).__name__,
                     backbone_module_file=sys.modules[type(plain.backbone).__module__].__file__)
    print("[S3] backbone repr equal %s | head repr equal %s | named tensors equal %s/%s | %s"
          % (r_bb, r_hd, s_bb, s_hd, rep["S3"]["backbone_module_file"]))
    if not (r_bb and r_hd and s_bb and s_hd):
        fails.append("S3 module tree differs")

    neg = None
    if a.neg:
        neg, _ = build_ptv3(device=dev, shuffle_orders=False, half=True, ckpt_path=a.neg)
        rep["neg"] = a.neg

    # ------------------------------------------------------------------ S4 behaviour
    os.environ["PTV3_CKPT"] = a.out
    os.environ["PTV3_HALF"] = "1"
    os.environ["PTV3_SHUFFLE"] = "0"
    os.environ["PTV3_GPU_VOXEL"] = "1"
    os.environ["PTV3_FAST_HILBERT"] = "1"
    import ptv3_worker
    worker = ptv3_worker.Segmenter()
    rep["worker"] = dict(ckpt=worker.ckpt, sha256=worker.ckpt_sha256, tensors=worker.ckpt_tensors,
                         half=worker.half, shuffle=worker.shuffle, gpu_voxel=worker.gpu_voxel,
                         fast_hilbert=worker.fast_hilbert)
    assert worker.ckpt == a.out and worker.ckpt_sha256 == rep["out_sha256"], "worker loaded another file"
    print("[worker] Segmenter loaded %s (sha %s, %d tensors)"
          % (worker.ckpt, worker.ckpt_sha256[:16], worker.ckpt_tensors))
    worker.warmup(1)

    SK = np.asarray(LS.sk_lut(), np.int32)
    NU = np.asarray(LS.nusc_lut(), np.int32)
    K = len(LS.COARSE)

    def make_input(pts):
        coord = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
        stren = np.ascontiguousarray(pts[:, 3:4] * INTENSITY_SCALE, dtype=np.float32)
        cv, sv, gv, inv = voxelize(coord, stren, GRID)
        c = torch.from_numpy(cv).to(dev); s = torch.from_numpy(sv).to(dev)
        g = torch.from_numpy(gv).to(dev)
        feat = torch.cat([c.half(), s.half()], 1)
        d = dict(coord=c.half(), grid_coord=g, feat=feat,
                 offset=torch.tensor([c.shape[0]], device=dev, dtype=torch.long))
        return d, inv

    @torch.inference_mode()
    def f_plain(m, d):
        return m(dict(d))["seg_logits"].float()

    @torch.inference_mode()
    def f_distil(d):
        return distil._logits16(dict(d)).float()

    STRATA = [("all", 0.0), ("margin>1", 1.0), ("margin>2", 2.0), ("margin>4", 4.0)]
    PAIRS = ["distil_vs_plain", "distil_vs_plain_2", "distil_vs_distil", "plain_vs_plain",
             "plain_vs_worker"] + (["plain_vs_NEG"] if neg is not None else [])
    agg = {p: {s: [0, 0] for s, _ in STRATA} for p in PAIRS}          # [n, n_agree]
    dl = {p: [] for p in PAIRS if p != "plain_vs_worker"}               # |delta logit| samples
    conf_ = {n: np.zeros((K, K + 1), np.int64) for n in ["distil", "distil_2", "plain", "plain_2", "worker"] + (["NEG"] if neg is not None else [])}
    hist16 = {n: np.zeros(16, np.int64) for n in conf_}
    rng = np.random.default_rng(0)

    def pair(name, la, lb):
        pa = la.argmax(1); pb = lb.argmax(1)
        t2 = la.topk(2, dim=1).values
        margin = t2[:, 0] - t2[:, 1]
        for s, thr in STRATA:
            m = margin > thr if thr > 0 else torch.ones_like(margin, dtype=torch.bool)
            agg[name][s][0] += int(m.sum()); agg[name][s][1] += int((pa[m] == pb[m]).sum())
        if name in dl:
            d = (la - lb).abs().max(1).values.cpu().numpy()
            dl[name].append(d[rng.choice(len(d), min(len(d), 20000), replace=False)])

    def score(name, lab_pts, gt):
        m = gt >= 0
        p = NU[lab_pts[m]]; g = gt[m]
        p = np.where(p == LS.UNMAPPED, K, p)
        conf_[name] += np.bincount(g * (K + 1) + p, minlength=K * (K + 1)).reshape(K, K + 1)
        hist16[name] += np.bincount(lab_pts, minlength=16)

    t0 = time.time()
    for f in frames:
        pts = read_scan(f)
        gt = SK[np.fromfile(LABELS % f, dtype=np.uint32) & 0xFFFF]
        assert len(gt) == len(pts)
        d, inv = make_input(pts)
        D1 = f_distil(d); D2 = f_distil(d)
        P1 = f_plain(plain, d); P2 = f_plain(plain, d)
        pair("distil_vs_plain", D1, P1); pair("distil_vs_plain_2", D2, P2)
        pair("distil_vs_distil", D1, D2); pair("plain_vs_plain", P1, P2)
        if neg is not None:
            N1 = f_plain(neg, d); pair("plain_vs_NEG", P1, N1)
            score("NEG", N1.argmax(1).cpu().numpy()[inv].astype(np.int64), gt)
        wl, _ = worker.segment(pts)                       # the deployed path, per point
        wl = wl.astype(np.int64)
        p1_pts = P1.argmax(1).cpu().numpy()[inv].astype(np.int64)
        d1_pts = D1.argmax(1).cpu().numpy()[inv].astype(np.int64)
        # per-point agreement plain-vs-worker (worker voxelises on the GPU, same first-in-cell rule)
        agg["plain_vs_worker"]["all"][0] += len(wl); agg["plain_vs_worker"]["all"][1] += int((wl == p1_pts).sum())
        score("distil", d1_pts, gt); score("plain", p1_pts, gt); score("worker", wl, gt)
        score("distil_2", D2.argmax(1).cpu().numpy()[inv].astype(np.int64), gt)
        score("plain_2", P2.argmax(1).cpu().numpy()[inv].astype(np.int64), gt)
        print("  frame %4d  voxels %6d  agree distil/plain %.4f  distil/distil %.4f  plain/plain %.4f  plain/worker(pts) %.4f"
              % (f, D1.shape[0],
                 float((D1.argmax(1) == P1.argmax(1)).float().mean()),
                 float((D1.argmax(1) == D2.argmax(1)).float().mean()),
                 float((P1.argmax(1) == P2.argmax(1)).float().mean()),
                 float((wl == p1_pts).mean())), flush=True)

    def metrics(C):
        row = C.sum(1); col = C[:, :K].sum(0); diag = np.diag(C[:, :K])
        iou = {LS.COARSE[k]: (100.0 * diag[k] / (row[k] + col[k] - diag[k]) if (row[k] + col[k] - diag[k]) else float("nan"))
               for k in range(K) if row[k] > 0}
        return dict(point_acc=100.0 * diag.sum() / max(1, C.sum()),
                    miou9=float(np.mean(list(iou.values()))), per_class_iou={k: round(v, 3) for k, v in iou.items()},
                    n_eval=int(C.sum()))

    S4 = dict(agreement={p: {s: dict(n=v[0], agree=(v[1] / v[0] if v[0] else None)) for s, v in d_.items()}
                         for p, d_ in agg.items()},
              abs_dlogit_max_per_voxel={p: dict(mean=float(np.mean(np.concatenate(v))),
                                                p50=float(np.percentile(np.concatenate(v), 50)),
                                                p99=float(np.percentile(np.concatenate(v), 99)),
                                                max=float(np.max(np.concatenate(v))))
                                        for p, v in dl.items() if v},
              gt_score={n: metrics(C) for n, C in conf_.items()},
              class_hist16={n: {NUSCENES_CLASSES[i]: int(h[i]) for i in range(16)} for n, h in hist16.items()},
              seconds=time.time() - t0)
    rep["S4"] = S4

    # VERDICT RULE (relative to the instrument's own noise, measured in the same run):
    #  (a) for every margin stratum the cross-model agreement must be >= the smaller of the two
    #      same-model repeat agreements minus 0.003;
    #  (b) the deployed path (per-point plain_vs_worker) must be >= the same-model "all" agreement
    #      minus 0.005;
    #  (c) the negative control, when given, must sit >= 0.02 below the cross-model "all" agreement;
    #  (d) point accuracy against GT of the plain and worker draws within 0.5 of the distil mean
    #      (the historical broken pairing sat 60 points below; the within-model spread is ~0.1).
    def ag(p, s):
        v = agg[p][s]; return v[1] / v[0] if v[0] else float("nan")
    vi = {}
    for s, _ in STRATA:
        cross = min(ag("distil_vs_plain", s), ag("distil_vs_plain_2", s))
        within = min(ag("distil_vs_distil", s), ag("plain_vs_plain", s))
        vi[s] = dict(cross=cross, within=within, ok=bool(cross >= within - 0.003))
        if not vi[s]["ok"]:
            fails.append("S4(a) %s: cross %.5f < within %.5f - 0.003" % (s, cross, within))
    w_all = ag("plain_vs_worker", "all"); within_all = min(ag("distil_vs_distil", "all"), ag("plain_vs_plain", "all"))
    vi["worker"] = dict(plain_vs_worker=w_all, within_all=within_all, ok=bool(w_all >= within_all - 0.005))
    if not vi["worker"]["ok"]:
        fails.append("S4(b) deployed path per-point agreement %.5f < %.5f - 0.005" % (w_all, within_all))
    if neg is not None:
        n_all = ag("plain_vs_NEG", "all"); c_all = ag("distil_vs_plain", "all")
        vi["neg"] = dict(neg_all=n_all, cross_all=c_all, separated=bool(n_all <= c_all - 0.02))
        if not vi["neg"]["separated"]:
            fails.append("S4(c) negative control not separated: %.5f vs %.5f" % (n_all, c_all))
    gs = S4["gt_score"]
    d_acc = 0.5 * (gs["distil"]["point_acc"] + gs["distil_2"]["point_acc"])
    for nme in ("plain", "plain_2", "worker"):
        if abs(gs[nme]["point_acc"] - d_acc) > 0.5:
            fails.append("S4(d) %s point acc %.3f vs distil mean %.3f" % (nme, gs[nme]["point_acc"], d_acc))
    vi["gt"] = dict(distil_mean_acc=d_acc,
                    within_model_acc_spread=max(abs(gs["distil"]["point_acc"] - gs["distil_2"]["point_acc"]),
                                                abs(gs["plain"]["point_acc"] - gs["plain_2"]["point_acc"])),
                    within_model_miou_spread=max(abs(gs["distil"]["miou9"] - gs["distil_2"]["miou9"]),
                                                 abs(gs["plain"]["miou9"] - gs["plain_2"]["miou9"])),
                    cross_model_miou_spread=max(abs(gs[x]["miou9"] - gs[y]["miou9"]) for x in ("distil", "distil_2") for y in ("plain", "plain_2", "worker")))
    S4["verdict_inputs"] = vi
    print("\n=== %s  S4 agreement (fraction of voxels with identical argmax) ===" % a.tag)
    for p in PAIRS:
        print("  %-20s " % p + "  ".join("%s %.5f (n=%d)" % (s, agg[p][s][1] / max(1, agg[p][s][0]), agg[p][s][0]) for s, _ in STRATA if agg[p][s][0]))
    for p, v in S4["abs_dlogit_max_per_voxel"].items():
        print("  |dlogit|max/voxel %-18s mean %.4f p50 %.4f p99 %.4f max %.4f" % (p, v["mean"], v["p50"], v["p99"], v["max"]))
    for n, m in S4["gt_score"].items():
        print("  GT common-9 (%d frames) %-7s point acc %.3f  mIoU-9 %.3f" % (len(frames), n, m["point_acc"], m["miou9"]))
    rep["fails"] = fails
    rep["PASS"] = not fails
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=2)
    print("\n%s  ->  %s" % ("PASS" if not fails else "FAIL: " + " | ".join(fails), a.report))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
