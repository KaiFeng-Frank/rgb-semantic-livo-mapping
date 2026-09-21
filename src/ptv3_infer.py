#!/usr/bin/env python3
# =============================================================================
#  DO NOT USE THIS MODULE FOR INFERENCE.  IT LOADS A BROKEN MODEL.
#
#  POINTCEPT_ROOT here points at src/Pointcept, which is Pointcept HEAD.  HEAD's
#  PTv3 is NOT the architecture the released nuScenes weights were trained on;
#  this file papers over the difference by renaming the config key
#  cls_mode -> enc_mode, which makes state_dict keys line up while the forward
#  pass differs.  MEASURED over 20 seq07 frames against SemanticKITTI GT:
#
#     this path (HEAD + rename) :  25.66 % point acc,  car IoU  0.00 %
#                                  (`car` predicted on 5 of 2.4 M points)
#     Pointcept v1.5.1, x1.0    :  57.48 % point acc,  car IoU 86.18 %
#     Pointcept v1.5.1, x0.2    :  86.34 % point acc,  car IoU 91.22 %
#
#  The intensity_scale = 1.0 default below was calibrated on the broken model
#  and is also wrong; the correct value is 0.2 (see CRITICAL_CONSTRAINTS.md).
#
#  USE INSTEAD:  src/ptv3_loader_verified.py (Pointcept v1.5.1, strict=True),
#  wrapped by src/ptv3_worker.py.  Only the pure label-mapping tables (COARSE,
#  NUSC16_TO_COARSE, SK_TO_COARSE) below are safe, and they are duplicated in
#  sem_core.py so they can be imported without torch.
# =============================================================================
"""
PTv3 semantic segmentation service for KITTI-style LiDAR scans.

Reusable module + CLI.

    from ptv3_infer import PTv3Segmenter
    seg = PTv3Segmenter()                 # loads nuScenes PTv3-m1 base onto cuda:0
    labels, conf = seg.segment(pts_Nx4)   # uint16 [N], float32 [N]

Checkpoint: Pointcept/PointTransformerV3 -> nuscenes-semseg-pt-v3m1-0-base
  in_channels=4, feat = concat(coord_xyz, strength), grid_size=0.05, 16 classes.

INTENSITY SCALING (established by source inspection, do not change blindly):
  Pointcept NuScenesDataset.get_data(): strength = points[:,3] / 255   (raw nuScenes is 0..255)
  Pointcept SemanticKITTIDataset.get_data(): strength = scan[:,-1]     (raw KITTI is already 0..1)
  Measured on KITTI 2011_09_30_drive_0027 frame 0: intensity min 0.0 max 0.99 median 0.31.
  => Both datasets feed the network strength in [0,1]. For KITTI we pass intensity AS-IS
     (INTENSITY_SCALE = 1.0). Dividing by 255 here would be a bug.
"""

import os
import sys
import time
import types
import argparse
from collections import OrderedDict

import numpy as np
import torch

# ----------------------------------------------------------------------------- paths
POINTCEPT_ROOT = os.environ.get("POINTCEPT_ROOT", "/data/livo_sem/src/Pointcept")
CKPT_DIR = "/data/livo_sem/weights/nuscenes-semseg-pt-v3m1-0-base"
DEFAULT_WEIGHTS = os.path.join(CKPT_DIR, "model", "model_best.pth")
DEFAULT_CONFIG = os.path.join(CKPT_DIR, "config.py")

NUSCENES16 = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface",
    "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]

INTENSITY_SCALE_KITTI = 1.0     # KITTI .bin intensity is already [0,1]
INTENSITY_SCALE_NUSCENES_RAW = 1.0 / 255.0


# ----------------------------------------------------------------------------- bootstrap
def _bootstrap(pointcept_root=POINTCEPT_ROOT):
    """Make Pointcept importable without its optional compiled CUDA extensions.

    pointops / pointgroup_ops are only used by training/eval hooks, never on the
    inference path, but pointcept.models.__init__ pulls them in transitively.
    """
    if pointcept_root not in sys.path:
        sys.path.insert(0, pointcept_root)
    for name in ("pointops", "pointgroup_ops"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            m.__ptv3_stub__ = True
            sys.modules[name] = m


def load_config(path=DEFAULT_CONFIG):
    """config.py from the HF checkpoint repo is plain python; exec it."""
    ns = {}
    with open(path, "r") as f:
        exec(compile(f.read(), path, "exec"), ns)
    return ns


# ----------------------------------------------------------------------------- segmenter
class PTv3Segmenter:
    def __init__(
        self,
        weights=DEFAULT_WEIGHTS,
        config=DEFAULT_CONFIG,
        device="cuda",
        enable_flash=True,
        grid_size=None,
        intensity_scale=INTENSITY_SCALE_KITTI,
        z_shift=0.0,
        amp=False,   # torch.autocast(fp16) + spconv implicit_gemm -> "can't find suitable
                     # algorithm" (fp16 activations vs fp32 weights). fp32 weights + the
                     # model's own bf16 flash-attention path is both correct and fast enough.
        deterministic=True,
        seed=0,
        voxel_backend="gpu",   # "gpu" (fast, exact) | "pointcept" (reference GridSample)
        half=True,             # fp16 weights: spconv is happy when weights AND feats are fp16
        shuffle_orders=False,  # config ships True; it randomises the serialisation order on
                               # EVERY forward pass, so two runs on one scan agree only ~78%.
                               # Off => bit-reproducible output, which a mapping node needs.
        pointcept_root=POINTCEPT_ROOT,
    ):
        _bootstrap(pointcept_root)
        from pointcept.models import build_model
        from pointcept.datasets.transform import GridSample

        cfg = load_config(config)
        model_cfg = cfg["model"]
        self.class_names = list(cfg.get("names", NUSCENES16))
        self.num_classes = int(model_cfg["num_classes"])
        self.grid_size = float(grid_size if grid_size is not None
                               else cfg["data"]["test"]["test_cfg"]["voxelize"]["grid_size"])

        model_cfg["backbone"]["enable_flash"] = bool(enable_flash)
        model_cfg["backbone"]["shuffle_orders"] = bool(shuffle_orders)
        self.enable_flash = bool(enable_flash)
        self.shuffle_orders = bool(shuffle_orders)

        # Pointcept HEAD renamed the PTv3 encoder-only flag cls_mode -> enc_mode.
        # The checkpoint's config.py predates that rename; drop/translate kwargs the
        # installed constructor does not accept (state_dict strictness below is the
        # real check that the architecture still matches).
        import inspect as _inspect
        from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
            PointTransformerV3 as _PTv3)
        _accepted = set(_inspect.signature(_PTv3.__init__).parameters)
        _renames = {"cls_mode": "enc_mode"}
        _bb = model_cfg["backbone"]
        for _old, _new in _renames.items():
            if _old in _bb and _old not in _accepted and _new in _accepted:
                _bb[_new] = _bb.pop(_old)
        self.dropped_backbone_kwargs = [
            k for k in list(_bb) if k != "type" and k not in _accepted]
        for k in self.dropped_backbone_kwargs:
            _bb.pop(k)

        self.device = torch.device(device)
        self.amp = bool(amp) and self.device.type == "cuda"
        self.intensity_scale = float(intensity_scale)
        self.z_shift = float(z_shift)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.voxel_backend = voxel_backend
        self.half = bool(half) and torch.device(device).type == "cuda"

        model = build_model(model_cfg)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        clean = OrderedDict()
        for k, v in sd.items():
            clean[k[7:] if k.startswith("module.") else k] = v
        missing, unexpected = model.load_state_dict(clean, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint/architecture mismatch: missing={list(missing)[:8]} "
                f"unexpected={list(unexpected)[:8]}")
        self.load_report = (list(missing), list(unexpected))
        model.eval().to(self.device)
        if self.half:
            model.half()
        self.model = model
        self.feat_dtype = torch.float16 if self.half else torch.float32

        # GridSample(mode="train") = one representative point per 5 cm voxel + the
        # exact inverse index back to every original point. This is the mapping the
        # network's own val pipeline uses; do NOT replace it with a kNN.
        self.grid_sample = GridSample(
            grid_size=self.grid_size,
            hash_type="fnv",
            mode="train",
            return_inverse=True,
            return_grid_coord=True,
        )
        self.last_num_voxels = 0

    # -------------------------------------------------------------- preprocessing
    def voxelize_gpu(self, points):
        """Same contract as Pointcept GridSample(mode="train", return_inverse=True,
        return_grid_coord=True): one representative point per grid cell plus an inverse
        index from every original point to its cell's row.

        The cell index is computed in numpy exactly as GridSample does (float64 divide,
        floor, subtract min) - CUDA's float64 divide disagrees with numpy's on a handful
        of points that sit exactly on a cell boundary, which silently merges ~14 of 82k
        cells.  The expensive part (sort / unique over 120k keys) runs on the GPU.

        Two benign differences from the reference implementation:
          * exact 3-D raveling instead of the FNV hash (no collisions at all);
          * the representative is the cell's first point in input order rather than a
            uniformly random one (GridSample's choice is arbitrary by construction).
        """
        dev = self.device
        pts_np = np.asarray(points, dtype=np.float32)
        coord_np = np.ascontiguousarray(pts_np[:, :3], dtype=np.float32)
        if self.z_shift != 0.0:
            coord_np = coord_np.copy()
            coord_np[:, 2] += self.z_shift

        gc = np.floor(coord_np / np.array(self.grid_size)).astype(np.int64)
        gc -= gc.min(0)
        ext = gc.max(0) + 1
        key = (gc[:, 0] * ext[1] + gc[:, 1]) * ext[2] + gc[:, 2]

        key_t = torch.from_numpy(key).to(dev, non_blocking=True)
        _, inverse = torch.unique(key_t, sorted=True, return_inverse=True)
        order = torch.argsort(inverse, stable=True)
        sc = inverse[order]
        first = torch.ones_like(sc, dtype=torch.bool)
        first[1:] = sc[1:] != sc[:-1]
        idx_unique = order[first]

        coord = torch.from_numpy(coord_np).to(dev, non_blocking=True)
        strength = torch.from_numpy(
            np.ascontiguousarray(pts_np[:, 3:4] * self.intensity_scale)).to(dev, non_blocking=True)
        gc_t = torch.from_numpy(gc).to(dev, non_blocking=True)
        return dict(coord=coord[idx_unique], strength=strength[idx_unique],
                    grid_coord=gc_t[idx_unique].int(), inverse=inverse)

    def voxelize(self, points):
        if self.voxel_backend == "gpu":
            return self.voxelize_gpu(points)
        return self.voxelize_cpu(points)

    def voxelize_cpu(self, points):
        pts = np.asarray(points, dtype=np.float32)
        assert pts.ndim == 2 and pts.shape[1] >= 4, "expect N x 4 (x,y,z,intensity)"
        coord = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
        if self.z_shift != 0.0:
            coord = coord.copy()
            coord[:, 2] += self.z_shift
        strength = np.ascontiguousarray(
            pts[:, 3:4] * self.intensity_scale, dtype=np.float32)
        if self.deterministic:
            np.random.seed(self.seed)
        return self.grid_sample(dict(coord=coord, strength=strength))

    # -------------------------------------------------------------- inference
    @torch.inference_mode()
    def segment(self, points, return_logits=False):
        """points: (N,4) float32 x,y,z,intensity -> (labels uint16 [N], conf float32 [N])"""
        d = self.voxelize(points)
        dev = self.device

        def _t(x, dtype):
            if isinstance(x, torch.Tensor):
                return x.to(device=dev, dtype=dtype)
            return torch.from_numpy(np.ascontiguousarray(x)).to(device=dev, dtype=dtype)

        coord = _t(d["coord"], self.feat_dtype)
        grid_coord = _t(d["grid_coord"], torch.int32)
        strength = _t(d["strength"], self.feat_dtype)
        feat = torch.cat([coord, strength], dim=1)
        offset = torch.tensor([coord.shape[0]], device=dev, dtype=torch.long)
        self.last_num_voxels = int(coord.shape[0])

        inp = dict(coord=coord, grid_coord=grid_coord, feat=feat, offset=offset)
        if self.amp:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = self.model(inp)["seg_logits"]
        else:
            logits = self.model(inp)["seg_logits"]

        prob = torch.softmax(logits.float(), dim=-1)
        conf_v, lab_v = prob.max(dim=-1)

        inverse = _t(d["inverse"], torch.int64)
        labels = lab_v[inverse].to(torch.int32).cpu().numpy().astype(np.uint16)
        conf = conf_v[inverse].cpu().numpy().astype(np.float32)
        if return_logits:
            inv_np = d["inverse"]
            if isinstance(inv_np, torch.Tensor):
                inv_np = inv_np.cpu().numpy()
            return labels, conf, logits.float().cpu().numpy(), inv_np
        return labels, conf

    def warmup(self, n_points=120000, iters=3):
        """First call costs ~750 ms (spconv algorithm autotuning + cuDNN); steady state is
        ~68 ms.  Call this once at node start-up so the first real scan is not late."""
        rng = np.random.default_rng(0)
        fake = np.empty((n_points, 4), np.float32)
        fake[:, :3] = rng.uniform(-40, 40, (n_points, 3))
        fake[:, 2] = rng.uniform(-2.0, 2.0, n_points)
        fake[:, 3] = rng.uniform(0, 1, n_points)
        for _ in range(iters):
            self.segment(fake)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def histogram(self, labels):
        cnt = np.bincount(np.asarray(labels, dtype=np.int64), minlength=self.num_classes)
        return [(self.class_names[i], int(cnt[i])) for i in range(self.num_classes)]


# ----------------------------------------------------------------------------- GT mapping
# Coarse common label space for nuScenes-16 vs SemanticKITTI-raw agreement.
COARSE = ["car", "bicycle", "motorcycle", "truck", "bus", "other_vehicle", "person",
          "road", "sidewalk", "other_flat", "terrain", "vegetation", "manmade"]
C = {n: i for i, n in enumerate(COARSE)}
IGNORE = -1

NUSC16_TO_COARSE = np.array([
    C["manmade"],        # 0  barrier          (nuScenes barrier ~ SK fence/other-structure)
    C["bicycle"],        # 1  bicycle
    C["bus"],            # 2  bus
    C["car"],            # 3  car
    C["other_vehicle"],  # 4  construction_vehicle
    C["motorcycle"],     # 5  motorcycle
    C["person"],         # 6  pedestrian
    C["manmade"],        # 7  traffic_cone
    C["other_vehicle"],  # 8  trailer
    C["truck"],          # 9  truck
    C["road"],           # 10 driveable_surface
    C["other_flat"],     # 11 other_flat
    C["sidewalk"],       # 12 sidewalk
    C["terrain"],        # 13 terrain
    C["manmade"],        # 14 manmade
    C["vegetation"],     # 15 vegetation
], dtype=np.int32)

# SemanticKITTI raw semantic id (lower 16 bits of .label) -> coarse
SK_TO_COARSE = {
    0: IGNORE,            # unlabeled
    1: IGNORE,            # outlier
    10: C["car"], 252: C["car"],
    11: C["bicycle"], 31: C["bicycle"], 253: C["bicycle"],      # bicyclist folded into bicycle
    13: C["bus"], 257: C["bus"],
    15: C["motorcycle"], 32: C["motorcycle"], 255: C["motorcycle"],
    16: C["other_vehicle"], 256: C["other_vehicle"],            # on-rails
    18: C["truck"], 258: C["truck"],
    20: C["other_vehicle"], 259: C["other_vehicle"],
    30: C["person"], 254: C["person"],
    40: C["road"], 44: C["road"], 60: C["road"],                # road, parking, lane-marking
    48: C["sidewalk"],
    49: C["other_flat"],
    50: C["manmade"],     # building
    51: C["manmade"],     # fence
    52: C["manmade"],     # other-structure
    70: C["vegetation"],
    71: C["vegetation"],  # trunk
    72: C["terrain"],
    80: C["manmade"],     # pole
    81: C["manmade"],     # traffic-sign
    99: IGNORE,           # other-object (no clean nuScenes counterpart)
}


def sk_labels_to_coarse(label_file):
    raw = np.fromfile(label_file, dtype=np.uint32)
    sem = (raw & 0xFFFF).astype(np.int32)
    lut = np.full(260, IGNORE, dtype=np.int32)
    for k, v in SK_TO_COARSE.items():
        lut[k] = v
    sem = np.clip(sem, 0, 259)
    return lut[sem]


# ----------------------------------------------------------------------------- CLI
def read_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="/data/livo_sem/data/raw/2011_09_30/"
                                      "2011_09_30_drive_0027_sync/velodyne_points/data/0000000000.bin")
    ap.add_argument("--scan-dir", default=None, help="bench over the first --bench scans here")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--gt", default=None, help=".label file aligned with --scan")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--eval-frames", type=int, default=0)
    ap.add_argument("--no-flash", action="store_true")
    ap.add_argument("--amp", action="store_true", help="enable autocast fp16 (breaks spconv)")
    ap.add_argument("--fp32", action="store_true", help="fp32 weights instead of fp16")
    ap.add_argument("--voxel-backend", default="gpu", choices=["gpu", "pointcept"])
    ap.add_argument("--z-shift", type=float, default=0.0)
    ap.add_argument("--intensity-scale", type=float, default=INTENSITY_SCALE_KITTI)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    seg = PTv3Segmenter(enable_flash=not args.no_flash, amp=args.amp,
                        z_shift=args.z_shift, intensity_scale=args.intensity_scale,
                        voxel_backend=args.voxel_backend, half=not args.fp32)
    miss, unexp = seg.load_report
    print(f"[load] state_dict strict OK (missing={len(miss)} unexpected={len(unexp)}) "
          f"dropped_backbone_kwargs={seg.dropped_backbone_kwargs} "
          f"grid_size={seg.grid_size} flash={seg.enable_flash} amp={seg.amp} "
          f"intensity_scale={seg.intensity_scale} z_shift={seg.z_shift} "
          f"half={seg.half} voxel_backend={seg.voxel_backend} "
          f"shuffle_orders={seg.shuffle_orders}")
    if miss[:5] or unexp[:5]:
        print("   missing[:5]", miss[:5], " unexpected[:5]", unexp[:5])

    pts = read_bin(args.scan)
    print(f"[scan] {args.scan}  N={pts.shape[0]}  "
          f"intensity[min={pts[:,3].min():.3f} max={pts[:,3].max():.3f} med={np.median(pts[:,3]):.3f}]")

    labels, conf = seg.segment(pts)
    print(f"[out] labels {labels.shape} {labels.dtype}  conf {conf.shape} {conf.dtype} "
          f"mean_conf={conf.mean():.4f}  voxels={seg.last_num_voxels}")
    print("[histogram] class                 count      pct")
    tot = labels.shape[0]
    for name, c in sorted(seg.histogram(labels), key=lambda x: -x[1]):
        print(f"            {name:<22s}{c:>8d}  {100.0*c/tot:6.2f}%")

    if args.save:
        np.savez(args.save, labels=labels, conf=conf, points=pts)
        print("[saved]", args.save)

    # ---- latency
    if args.bench:
        import glob
        if args.scan_dir:
            files = sorted(glob.glob(os.path.join(args.scan_dir, "*.bin")))[:args.bench]
        else:
            files = [args.scan] * args.bench
        scans = [read_bin(f) for f in files]
        for _ in range(5):
            seg.segment(scans[0])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for s in scans:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            seg.segment(s)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        print(f"[bench] n={len(ts)} scans  mean={ts.mean():.1f} ms  "
              f"p50={np.percentile(ts,50):.1f}  p95={np.percentile(ts,95):.1f}  "
              f"max={ts.max():.1f}  min={ts.min():.1f}")
        print(f"[vram]  peak_alloc={torch.cuda.max_memory_allocated()/2**20:.0f} MiB  "
              f"peak_reserved={torch.cuda.max_memory_reserved()/2**20:.0f} MiB")

    # ---- GT agreement
    gt_files = []
    if args.gt:
        gt_files = [(args.scan, args.gt)]
    elif args.gt_dir and args.eval_frames:
        import glob
        sd = args.scan_dir or os.path.dirname(args.scan)
        sc = sorted(glob.glob(os.path.join(sd, "*.bin")))[:args.eval_frames]
        gl = sorted(glob.glob(os.path.join(args.gt_dir, "*.label")))[:args.eval_frames]
        gt_files = list(zip(sc, gl))
    if gt_files:
        K = len(COARSE)
        inter = np.zeros(K, np.int64); pcnt = np.zeros(K, np.int64); gcnt = np.zeros(K, np.int64)
        ok = 0; tot2 = 0
        for sp, gp in gt_files:
            p = read_bin(sp)
            gt = sk_labels_to_coarse(gp)
            assert gt.shape[0] == p.shape[0], f"{sp} {p.shape} vs {gp} {gt.shape}"
            lb, _ = seg.segment(p)
            pr = NUSC16_TO_COARSE[lb.astype(np.int64)]
            m = gt != IGNORE
            ok += int((pr[m] == gt[m]).sum()); tot2 += int(m.sum())
            for k in range(K):
                pk = (pr == k) & m; gk = (gt == k) & m
                inter[k] += int((pk & gk).sum()); pcnt[k] += int(pk.sum()); gcnt[k] += int(gk.sum())
        print(f"\n[GT agreement] frames={len(gt_files)}  overall point accuracy "
              f"= {100.0*ok/max(tot2,1):.2f}%  ({ok}/{tot2} labelled points)")
        print("  coarse class        GT pts    pred pts    correct    recall     IoU")
        ious = []
        for k in range(K):
            if gcnt[k] == 0 and pcnt[k] == 0:
                continue
            u = pcnt[k] + gcnt[k] - inter[k]
            iou = inter[k] / u if u else float("nan")
            rec = inter[k] / gcnt[k] if gcnt[k] else float("nan")
            if gcnt[k] > 0:
                ious.append(iou)
            print(f"  {COARSE[k]:<16s}{gcnt[k]:>9d}{pcnt[k]:>12d}{inter[k]:>11d}"
                  f"{100*rec:>9.2f}%{100*iou:>8.2f}%")
        print(f"  mIoU over {len(ious)} classes present in GT = {100*np.nanmean(ious):.2f}%")


if __name__ == "__main__":
    main()
