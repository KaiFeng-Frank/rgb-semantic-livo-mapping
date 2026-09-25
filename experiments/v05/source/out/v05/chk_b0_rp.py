import json, numpy as np
ref = {"B0": [r["3d"] for r in json.load(open("out/v04/arms/score_B0.json"))["repeats"]],
       "RP": [r["3d"] for r in json.load(open("out/v04/nokl_20260923/Rprime_noKL/peer_score_Rprime_noKL.json"))["repeats"]]}
for key in ("B0", "RP"):
    print("=== %s" % key)
    for r in (1, 2, 3):
        d = json.load(open("out/v05/map_%s_r%d.json" % (key, r)))
        line = "  r%d  " % r
        for s in ("outside", "frustum", "global"):
            line += "%s: replay %.3f/%.3f harness %.3f/%.3f | " % (s[:3], d["offline_all"][s]["acc_abstain_excluded"], d["offline_all"][s]["miou9_abstain_excluded"], ref[key][r-1][s]["acc_abstain_excluded"], ref[key][r-1][s]["miou_all_abstain_excluded"])
        print(line)
        print("       map_all_lookup out %.3f/%.3f  in %.3f/%.3f  glo %.3f/%.3f   (pointgt_inserted out mIoU %.3f, majority %.3f)" % (
            d["map_all_lookup"]["outside"]["acc_abstain_wrong"], d["map_all_lookup"]["outside"]["miou9_abstain_wrong"],
            d["map_all_lookup"]["frustum"]["acc_abstain_wrong"], d["map_all_lookup"]["frustum"]["miou9_abstain_wrong"],
            d["map_all_lookup"]["global"]["acc_abstain_wrong"], d["map_all_lookup"]["global"]["miou9_abstain_wrong"],
            d["map_pointgt_inserted"]["outside"]["miou9_abstain_excluded"], d["map_majority_inserted"]["outside"]["miou9_abstain_excluded"]))
print("(RP harness rows = the PEER machine's three draws; the local caches are new draws of the same checkpoint)")
