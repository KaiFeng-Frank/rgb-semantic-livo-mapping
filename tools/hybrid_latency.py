#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 7 -- assemble the latency table, including the hybrid's honest bracket.

The hybrid needs BOTH networks, so its cost is a bracket, not a number:
    serial    one GPU, the two forwards run one after the other
    parallel  two streams / two devices, the two forwards overlap
Both are computed PER FRAME from the raw series and only then reduced, because
max(p95_a, p95_b) is not the p95 of the per-frame max.

Stage accounting, stated so a reader can re-derive it:
    shared      2d.read_image_and_scan   (reads the scan AND the image; the 3D arm's
                                          own read_scan is a subset of it)
    3D work     3d.total_excl_read       (voxelise + forward + devoxelise)
    2D work     2d.total_excl_read       (resize/normalise + forward + argmax +
                                          project + z-buffer + per-point sample)
    serial   = shared + 3D work + 2D work
    parallel = shared + max(3D work, 2D work)
"""
import json, sys, argparse
import numpy as np


def stats(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)), max=float(v.max()), n=int(v.size))


def fmt(s):
    return "%.1f / %.1f / %.1f" % (s["mean"], s["p95"], s["max"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3d", required=True)
    ap.add_argument("--t2d", required=True, help="the 2D arm used in the hybrid")
    ap.add_argument("--extra", nargs="*", default=[], help="label=path more 2D rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    T3 = json.load(open(a.t3d))
    T2 = json.load(open(a.t2d))
    assert T3["frame_ids"] == T2["frame_ids"], "the two arms were timed on different frames"

    w3 = np.asarray(T3["raw_ms"]["total_excl_read"], float)
    w2 = np.asarray(T2["raw_ms"]["total_excl_read"], float)
    sh = np.asarray(T2["raw_ms"]["read_image_and_scan"], float)

    res = {
        "conditions": {
            "hardware": T3["hardware"], "precision": T3["precision"],
            "frames": T3["frames"], "seq": T3["seq"], "batch": 1,
            "3d": {"model": T3["model"], "points": T3["info"].get("points"),
                   "voxels": T3["info"].get("voxels"),
                   "grid_size": T3.get("grid_size"), "tta": "none"},
            "2d": {"model": T2.get("model"), "repo": T2.get("repo"),
                   "input_scale": T2.get("input_scale"), "scale_key": T2.get("scale_key"),
                   "tta": T2.get("tta"), "input_hw": T2.get("input_hw")},
            "note": "exclusive GPU; warm-up excluded; torch.cuda.synchronize around "
                    "every timed stage",
        },
        "arms": {
            "3d_total": stats(np.asarray(T3["raw_ms"]["total"], float)),
            "3d_work_excl_read": stats(w3),
            "2d_total": stats(np.asarray(T2["raw_ms"]["total"], float)),
            "2d_work_excl_read": stats(w2),
            "hybrid_serial": stats(sh + w3 + w2),
            "hybrid_parallel": stats(sh + np.maximum(w3, w2)),
        },
        "stages_3d": T3["stages_ms"],
        "stages_2d": T2["stages_ms"],
        "extra_2d": {},
    }
    for e in a.extra:
        lbl, path = e.split("=", 1)
        t = json.load(open(path))
        res["extra_2d"][lbl] = {
            "model": t.get("model"), "scale": t.get("input_scale"), "tta": t.get("tta"),
            "input_hw": t.get("input_hw"),
            "total": stats(np.asarray(t["raw_ms"]["total"], float)),
            "work_excl_read": stats(np.asarray(t["raw_ms"]["total_excl_read"], float)),
            "forward_only": stats(np.asarray(t["raw_ms"]["forward_incl_pre_post"], float)),
        }
    json.dump(res, open(a.out, "w"), indent=2)

    print("| arm | mean / p95 / max  ms | Hz (1/mean) |")
    print("|---|---|---|")
    for k in ("3d_total", "2d_total", "hybrid_serial", "hybrid_parallel"):
        s = res["arms"][k]
        print("| %-16s | %-20s | %.1f |" % (k, fmt(s), 1000.0 / s["mean"]))
    for lbl, d in res["extra_2d"].items():
        print("| %-16s | %-20s | %.1f |" % (lbl, fmt(d["total"]), 1000.0 / d["total"]["mean"]))
    print()
    print("3D stages:", json.dumps({k: round(v["mean"], 1) for k, v in T3["stages_ms"].items()}))
    print("2D stages:", json.dumps({k: round(v["mean"], 1) for k, v in T2["stages_ms"].items()}))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
