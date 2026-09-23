#!/usr/bin/env python3
"""Validate the archived v0.4 evidence and regenerate its completed-results table.

Uses only the standard library; no dataset, GPU, network or training is needed.
The historical verdict labels are retained in the archive, not adopted as claims.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/v04"
ARMS = ("B1", "B0", "D", "R", "Rprime", "C")
SUBSETS = ("outside", "frustum", "global")
LABELS = {
    "A": "Zero-shot",
    "B1": "Camera pseudo-labels, head only",
    "B0": "Camera pseudo-labels, full fine-tuning",
    "D": "GT, camera frustum",
    "R": "GT, random raw points",
    "Rprime": "GT, random surviving voxels",
    "C": "GT, all points + KL",
}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
            f"{label}: {actual} != {expected}")


def stats(values):
    require(len(values) >= 2 and all(math.isfinite(v) for v in values),
            "Missing or nonfinite inference draws")
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), values=values)


def check_stats(actual, expected, label):
    for key in ("mean", "sd"):
        close(actual[key], expected[key], f"{label}/{key}")
    require(actual["values"] == expected["values"], f"{label}: draw mismatch")


def config_seed(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "seed" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing seed: {path}")


def verify_snapshot():
    manifest = read(RESULTS / "snapshot_manifest.json")
    for entry in manifest["files"]:
        path = ROOT / entry["published"]
        require(path.resolve().is_relative_to(ROOT), "Invalid manifest path")
        data = path.read_bytes()
        require(len(data) == entry["bytes"], f"Size mismatch: {path}")
        require(hashlib.sha256(data).hexdigest() == entry["sha256"],
                f"SHA-256 mismatch: {path}")
    directory = RESULTS / "nokl_20260923"
    gate = read(directory / "pair_gate.json")
    require(gate["pass"] and all(gate["checks"].values()), "No-KL preflight gate failed")
    for arm, digest in gate["preflight_sha256"].items():
        path = directory / arm / "preflight.json"
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f"No-KL preflight hash mismatch: {arm}")
        preflight = read(path)
        require(preflight["pass"] and preflight["anchor_was_not_called"],
                f"No-KL anchor check failed: {arm}")
        require(hashlib.sha256((directory / "protocol.md").read_bytes()).hexdigest()
                == preflight["protocol_sha256"], "No-KL protocol hash mismatch")
        for relative, expected in preflight["shared_sha256"].items():
            archived = (RESULTS / "training/rare_weights.json" if relative == "out/pseudo/rare_weights.json"
                        else ROOT / "experiments/v04/source" / relative)
            require(hashlib.sha256(archived.read_bytes()).hexdigest() == expected,
                    f"Pinned source hash mismatch: {relative}")


def verify_nokl_final():
    directory = RESULTS / "nokl_20260923"
    manifest = read(directory / "final_manifest.json")
    for entry in manifest["files"]:
        path = directory / entry["path"]
        require(path.is_file(), f"Missing final no-KL artifact: {path}")
        data = path.read_bytes()
        require(len(data) == entry["bytes"], f"Final no-KL size mismatch: {path}")
        require(hashlib.sha256(data).hexdigest() == entry["sha256"],
                f"Final no-KL SHA-256 mismatch: {path}")
    comparison = read(directory / "comparison.json")
    require(comparison["count_match"]["passed"], "Final no-KL count gate failed")
    require(comparison["count_match"]["ratio"] == 1.0,
            "Final no-KL count ratio is not exactly one")
    require(comparison["means"]["D_noKL"] == 81.43809018658541,
            "Unexpected D_noKL primary metric")
    require(comparison["means"]["Rprime_noKL"] == 84.89932230993442,
            "Unexpected Rprime_noKL primary metric")
    for arm in ("D_noKL", "Rprime_noKL"):
        status = read(directory / arm / "status.json")
        audit = read(directory / arm / "completion_audit.json")
        require(status["phase"] == "complete", f"No-KL arm not complete: {arm}")
        require(len(audit["epochs"]) == 10 and audit["kl_points_all_zero"],
                f"No-KL audit incomplete: {arm}")


def build_summary():
    verify_snapshot()
    verify_nokl_final()
    baseline = read(RESULTS / "armA_spread_pooled.json")
    require(baseline["n_draws"] == 6, "Expected six baseline inference draws")
    a = dict(label=LABELS["A"], training_runs=0, inference_draws=6, subsets={})
    for subset in SUBSETS:
        b = baseline["subsets"][subset]
        m9 = stats(b["miou9"]["values"])
        check_stats(m9, b["miou9"], "A/" + subset)
        classes = b["per_class_iou"]
        m8 = stats([statistics.mean(row["values"][i] for name, row in classes.items()
                                    if name != "two_wheeler") for i in range(6)])
        a["subsets"][subset] = dict(n_eval=b["n_eval"], miou9=m9, miou8=m8,
            delta_miou9=0.0, delta_miou8=0.0,
            per_class={name: row["mean"] for name, row in classes.items()})
    arms = {"A": a}
    audits = {}
    for arm in ARMS:
        scorer = read(RESULTS / f"arms/score_{arm}.json")
        verdict = read(RESULTS / f"arms/verdict_{arm}.json")
        require(scorer["protocol"]["n_frames"] == 1101, f"Incomplete test: {arm}")
        require(scorer["protocol"]["frames_first_last"] == [0, 1100], f"Wrong frames: {arm}")
        repeats = [draw["3d"] for draw in scorer["repeats"]]
        require(len(repeats) == verdict["n_draws"] == 3, f"Missing draws: {arm}")
        epochs = [json.loads(line) for line in
                  (RESULTS / f"training/{arm}/rare_class_audit.jsonl").read_text().splitlines()]
        require([row["epoch"] for row in epochs] == list(range(1, 11)),
                f"Incomplete or duplicate epochs: {arm}")
        for row in epochs:
            require(sum(row["per_class"].values()) == row["supervised_points"] > 0,
                    f"Supervision count mismatch: {arm}, epoch {row['epoch']}")
        lambdas = {row["kl_lambda"] for row in epochs}
        require(len(lambdas) == 1, f"KL coefficient changed during training: {arm}")
        audits[arm] = epochs
        result = dict(label=LABELS[arm], training_runs=1, inference_draws=3,
                      seed=config_seed(RESULTS / f"training/{arm}/config.py"),
                      epochs_completed=10, kl_lambda=lambdas.pop(),
                      historical_p3=verdict["p3"], subsets={})
        for subset in SUBSETS:
            old = verdict["subsets"][subset]
            expected_count = a["subsets"][subset]["n_eval"]
            for draw in repeats:
                require(draw[subset]["n_eval"] == expected_count, f"Changed evaluated set: {arm}")
                require(draw[subset]["n_abstained"] == 0, f"Unexpected abstentions: {arm}")
            m9 = stats([draw[subset]["miou_nine_abstain_excluded"] for draw in repeats])
            m8 = stats([statistics.mean(value for name, value in
                        draw[subset]["per_class_iou_abstain_excluded"].items()
                        if name != "two_wheeler") for draw in repeats])
            check_stats(m9, old["miou9_arm"], arm + "/" + subset + "/m9")
            # Historical mIoU-8 uses the scorer's rounded per-class IoUs.
            close(m8["mean"], old["miou8_arm"]["mean"], arm + "/" + subset + "/m8")
            delta9 = m9["mean"] - a["subsets"][subset]["miou9"]["mean"]
            delta8 = m8["mean"] - a["subsets"][subset]["miou8"]["mean"]
            close(delta9, old["delta_miou9"], arm + "/delta9")
            close(delta8, old["delta_miou8"], arm + "/delta8")
            per_class = {name: statistics.mean(draw[subset]["per_class_iou_abstain_excluded"][name]
                         for draw in repeats) for name in old["per_class"]}
            for name, value in per_class.items():
                close(value, old["per_class"][name]["arm"], arm + "/" + name)
            result["subsets"][subset] = dict(n_eval=expected_count, miou9=m9, miou8=m8,
                delta_miou9=delta9, delta_miou8=delta8, per_class=per_class)
        arms[arm] = result
    d = [row["supervised_points"] for row in audits["D"]]
    rp = [row["supervised_points"] for row in audits["Rprime"]]
    mean, sd = statistics.mean(d), statistics.stdev(d)
    lo, hi = mean - 5 * sd, mean + 5 * sd
    count = dict(D_epochs=d, Rprime_epochs=rp, D_mean=mean, D_sd=sd,
                 interval=[lo, hi], ratio=statistics.mean(rp) / mean,
                 passed=all(lo <= value <= hi for value in rp))
    require(count["passed"], "Rprime whole-stream count gate failed")
    calibration = read(RESULTS / "calibration/metrics.json")
    require(calibration["go"] and all(calibration["checks"].values()), "Host calibration failed")
    primary = lambda arm: arms[arm]["subsets"]["outside"]["miou9"]["mean"]
    return dict(metric="SemanticKITTI seq07 outside-frustum common-9 mIoU (%)",
                frames=1101, uncertainty="Sample SD across inference repeats, not training runs",
                selection="seq08 GT validation", arms=arms,
                Rprime_count_gate=count,
                contrasts=dict(B0_minus_A=primary("B0")-primary("A"),
                               Rprime_minus_D=primary("Rprime")-primary("D"),
                               C_minus_Rprime=primary("C")-primary("Rprime")))


def render(summary):
    lines = ["# Completed v0.4 runs", "",
        "Generated by `python3 tools/summarize_v04.py --write` from the archived scorer outputs.",
        "All values are percentages; deltas are percentage points. ± is sample SD over inference draws.",
        "Each adapted arm is one training run with three inference draws; A has six inference draws.", "",
        "| Arm | Supervision | Outside mIoU-9 ± SD | Δ vs A | Outside mIoU-8 | Δ8 vs A | In-frustum mIoU-9 | Global mIoU-9 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm, row in summary["arms"].items():
        out = row["subsets"]["outside"]
        lines.append(f"| {arm} | {row['label']} | {out['miou9']['mean']:.2f} ± {out['miou9']['sd']:.2f} "
                     f"| {out['delta_miou9']:+.2f} | {out['miou8']['mean']:.2f} | {out['delta_miou8']:+.2f} "
                     f"| {row['subsets']['frustum']['miou9']['mean']:.2f} | {row['subsets']['global']['miou9']['mean']:.2f} |")
    lines += ["", "mIoU-8 excludes `two_wheeler` and is a decomposition only. D, R, Rprime and C use target 3D GT for training.",
              "The historical B0 sign test matches 5/8 cells; its +7.96 score gain does not validate that mechanism story.", "",
              "## Training record", "", "| Arm | Seed | Epochs | Realised KL coefficient |",
              "|---|---:|---:|---:|"]
    for arm in ARMS:
        row = summary["arms"][arm]
        lines.append(f"| {arm} | {row['seed']} | {row['epochs_completed']} | {row['kl_lambda']:.9f} |")
    count = summary["Rprime_count_gate"]
    lines += ["", f"Rprime/D mean supervised-voxel ratio: **{count['ratio']:.6f}**. "
              "All 10 Rprime epoch totals fall within D's mean ± 5 sample SD.",
              "This matches total supervision count; class composition and the geometry of the KL region still differ.", "",
              "## Outside-frustum class IoU", "",
              "| Class | A | B1 | B0 | D | R | Rprime | C |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in summary["arms"]["A"]["subsets"]["outside"]["per_class"]:
        values = [row["subsets"]["outside"]["per_class"][name] for row in summary["arms"].values()]
        lines.append("| " + name + " | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", "`two_wheeler` remains an abstain cell for the preregistered per-class sign interpretation.",
              "See [interpretation and next steps](../../docs/v04_results.md) before drawing causal conclusions.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="regenerate the JSON summary and Markdown table")
    mode.add_argument("--check", action="store_true", help="verify the archive and checked-in generated outputs")
    args = parser.parse_args()
    summary = build_summary()
    outputs = {"completed_summary.json": json.dumps(summary, indent=2) + "\n",
               "completed_summary.md": render(summary)}
    for name, content in outputs.items():
        if args.write:
            (RESULTS / name).write_text(content)
        elif args.check:
            require((RESULTS / name).read_text() == content, f"Stale generated output: {name}")
    if args.check:
        print("PASS: hashes, preflights, 6 completed training audits, 18 inference draws, metric tables, voxel-count gate and final no-KL pair")
    elif args.write:
        print("Wrote results/v04/completed_summary.{json,md}")
    else:
        print(outputs["completed_summary.md"], end="")


if __name__ == "__main__":
    main()
