#!/usr/bin/env python3
"""tools/sparse_diag_confirm.py -- executes amendment A2 of out/v06_sparsediag/PREREG.md."""
import json
import os
import sys

O = "/data/livo_sem/out/v06_sparsediag"
TAGS = ["B1_r1", "D_r1", "C_r1", "R_r1", "RP_r1"]
print("=" * 80)
print("A2 CONFIRMATION -- CONTEXT on arms not looked at before the threshold was chosen")
passes = 0; done = 0
for t in TAGS:
    p = "%s/diag_%s.json" % (O, t)
    if not os.path.exists(p):
        print("  %-6s missing" % t); continue
    j = json.load(open(p)); d = j["sparse_diag"]
    acc = d["isolation"]["outside"]["accounting_n_equal"] and d["isolation"]["outside"]["accounting_correct_equal"]
    cur = d["isolation"]["outside"]["curve"]
    lo = [b for b in cur if b["n"] and b["hi"] <= 0.5 + 1e-9]
    hi = [b for b in cur if b["n"] and b["lo"] >= 0.5 - 1e-9]
    a_lo = sum(b["n"] * b["acc"] for b in lo) / sum(b["n"] for b in lo)
    a_hi = sum(b["n"] * b["acc"] for b in hi) / sum(b["n"] for b in hi)
    n_hi = sum(b["n"] for b in hi)
    w = d["populations"]["global"]["sparse_wrong"]; c = d["populations"]["global"]["sparse_correct"]
    i_ = (a_lo - a_hi) >= 5.0
    ii = w["nonsplit_d8_min_median"] > c["d8_min_q"][1]
    ok = i_ and ii
    done += 1; passes += ok
    print("  %-6s acct %-5s | acc d8<0.5 %.2f%%  d8>=0.5 %.2f%% (%d pts)  (i') %-5s | d8min wrong %.3f vs correct %.3f (ii) %-5s -> %s"
          % (t, acc, a_lo, a_hi, n_hi, i_, w["nonsplit_d8_min_median"], c["d8_min_q"][1], ii, "PASS" if ok else "fail"))
    print("         split %.1f%%  camera-reachable(non-split) %.1f%%  of %d sparse wrong cells"
          % (100.0 * w["split"] / w["cells"], 100.0 * w["nonsplit_camera_labelable"] / max(1, w["nonsplit"]), w["cells"]))
print()
if done < len(TAGS):
    print("  -> INCOMPLETE (%d/%d arms)" % (done, len(TAGS))); sys.exit(1)
print("  -> CONTEXT %s  (%d of %d arms pass; >= 4 required)" % ("CONFIRMED" if passes >= 4 else "NOT CONFIRMED", passes, len(TAGS)))
