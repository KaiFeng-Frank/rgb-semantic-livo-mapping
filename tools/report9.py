#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report9.py -- turn score_2d_vs_3d.py's JSON into the citable markdown tables."""
import sys, json, argparse
import numpy as np

ARM_LABEL = {
    "3d": "**A. 3D** PTv3 on the cloud",
    "2d": "**B. 2D** Cityscapes projected",
    "2d_naive_projection": "B'. 2D, naive projection (no occlusion test)",
    "hybrid": "**C. Hybrid** 2D in frustum, 3D outside",
}


def G(agg, arm, sub, met):
    try:
        v = agg[arm][sub][met]
    except KeyError:
        return None
    return v if isinstance(v, dict) else {"mean": v, "half_range": 0.0}


def F(g, nd=2):
    if g is None or g["mean"] != g["mean"]:
        return "--"
    h = g.get("half_range", 0.0)
    return ("%.*f" % (nd, g["mean"])) + (" ±%.*f" % (nd, h) if h > 1e-12 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--arms", default="3d,2d,hybrid,2d_naive_projection")
    ap.add_argument("--subset", default="nine")
    a = ap.parse_args()
    R = json.load(open(a.results))
    agg, P = R["agg"], R["protocol"]
    arms = [x for x in a.arms.split(",") if x in agg]
    S = a.subset
    out = []
    w = out.append

    w("### Conditions")
    w("")
    w("- Sequence: %s, %d frames (`%s`)" % (P["sequence"], P["n_frames"], P["frames_spec"]))
    w("- Label space: %s -- %s" % (P["label_space"]["name"], ", ".join(P["label_space"]["classes"])))
    w("- Repeats: %s" % P["repeats"])
    w("- Every number is `mean ±half-range` over the repeats. A gap smaller than the")
    w("  half-range is NOT a difference.")
    w("")

    # ---------------- headline ------------------------------------------ #
    w("### Table 1 -- the citable table")
    w("")
    rows = []
    for arm in arms:
        gl, fr = "global", "frustum"
        rows.append([
            ARM_LABEL.get(arm, arm),
            F(G(agg, arm, gl, "coverage"), 4),
            F(G(agg, arm, fr, "acc_abstain_excluded")),
            F(G(agg, arm, fr, "miou_%s_abstain_excluded" % S)),
            F(G(agg, arm, gl, "acc_abstain_excluded")),
            F(G(agg, arm, gl, "miou_%s_abstain_excluded" % S)),
            F(G(agg, arm, gl, "acc_abstain_wrong")),
            F(G(agg, arm, gl, "miou_%s_abstain_wrong" % S)),
        ])
    head = ["arm", "coverage", "in-frustum acc %", "in-frustum mIoU %",
            "global acc % (ii)", "global mIoU % (ii)",
            "global acc % (i)", "global mIoU % (i)"]
    w("| " + " | ".join(head) + " |")
    w("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        w("| " + " | ".join(r) + " |")
    w("")
    w("(i) = unlabelled counted WRONG; (ii) = unlabelled ABSTAINED (excluded from the")
    w("denominator). In-frustum columns are over the shared mask (in frustum AND GT not")
    w("EXCLUDED); both arms see the identical point set there.")
    w("")

    # ---------------- boundary / interior -------------------------------- #
    w("### Table 2 -- boundary leakage and structure")
    w("")
    subs = ["sem_boundary", "sem_interior", "depth_edge", "depth_interior"]
    head = ["arm"] + sum([["%s acc %%" % s, "%s mIoU %%" % s] for s in subs], [])
    w("| " + " | ".join(head) + " |")
    w("|" + "|".join(["---"] * len(head)) + "|")
    for arm in arms:
        r = [ARM_LABEL.get(arm, arm)]
        for s in subs:
            r += [F(G(agg, arm, s, "acc_abstain_excluded")),
                  F(G(agg, arm, s, "miou_%s_abstain_excluded" % S))]
        w("| " + " | ".join(r) + " |")
    w("")

    # ---------------- range ---------------------------------------------- #
    for pref, title in (("frustum_range", "in-frustum"), ("range", "global")):
        w("### Table 3%s -- accuracy vs range, %s" % ("a" if pref == "frustum_range" else "b", title))
        w("")
        tags = ["0_10", "10_20", "20_30", "30_50", "50_inf"]
        head = ["arm / convention"] + [t.replace("_", "-").replace("-inf", "+") + " m" for t in tags]
        w("| " + " | ".join(head) + " |")
        w("|" + "|".join(["---"] * len(head)) + "|")
        for arm in arms:
            for conv, lbl in (("acc_abstain_excluded", "(ii)"), ("acc_abstain_wrong", "(i)")):
                r = ["%s %s" % (ARM_LABEL.get(arm, arm), lbl)]
                for t in tags:
                    r.append(F(G(agg, arm, "%s_%s" % (pref, t), conv)))
                w("| " + " | ".join(r) + " |")
        w("")
        gt = ["| GT points |"]
        w("| GT points | " + " | ".join(
            "%.1f %%" % (100.0 * G(agg, arms[0], "%s_%s" % (pref, t), "n_eval")["mean"] /
                         G(agg, arms[0], "global" if pref == "range" else "frustum", "n_eval")["mean"])
            for t in tags) + " |")
        w("")

    # ---------------- per class ------------------------------------------ #
    w("### Table 4 -- per-class IoU, in frustum (repeat 0; convention (ii))")
    w("")
    cls = P["label_space"]["classes"]
    w("| arm | " + " | ".join(cls) + " |")
    w("|" + "|".join(["---"] * (len(cls) + 1)) + "|")
    for arm in arms:
        d = agg[arm]["frustum"].get("per_class_iou_abstain_excluded", {})
        w("| " + ARM_LABEL.get(arm, arm) + " | " +
          " | ".join(("%.2f" % d[c]) if c in d else "--" for c in cls) + " |")
    w("")

    # ---------------- diagnostics ---------------------------------------- #
    d = R["repeats"][0].get("2d", {}).get("_diag", {})
    if d:
        w("### 2D-arm diagnostics (repeat 0)")
        w("")
        for k, v in d.items():
            w("- `%s` = %s" % (k, v))
        w("")
    w("### Concessions made to the 2D arm")
    w("")
    for c in P["pro_2d_concessions"]:
        w("- " + c)
    print("\n".join(out))


if __name__ == "__main__":
    main()
