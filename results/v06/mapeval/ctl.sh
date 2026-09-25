cd /data/livo_sem
P=/data/miniconda3/envs/ags/bin/python
$P - << "PY" > out/v06_mapeval/ctl_ast.txt 2>&1
import ast
tree = ast.parse(open("tools/map_eval.py").read())
main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
stored = {x.id for x in ast.walk(main) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
mods = {"S","LS","np","json","os","sys","time","seqreg","RAW","LB","K","NAMES","UNM","ABST","NU","SK","synth","parse_ts","Conf","KeyIndex"}
print("module names STORED inside main():", sorted(stored & mods) or "none")
PY
$P tools/map_eval.py --seq 07 --pred out/v04/arms/B0_r1 --pred-order raw --tag ctl_B0_r1 \
    --json-out out/v06_mapeval/ctl_B0_r1.json > logs/mapeval_ctl_B0_r1.log 2>&1
echo "rc=$?" >> out/v06_mapeval/ctl_ast.txt
$P - << "PY" >> out/v06_mapeval/ctl_ast.txt 2>&1
import json
n = json.load(open("out/v06_mapeval/ctl_B0_r1.json"))
print("map_all_lookup == v0.5            :", n["map_all_lookup"] == json.load(open("out/v05/map_B0_r1.json"))["map_all_lookup"])
print("point anatomy == preregistered run:", n["anatomy"] == json.load(open("out/v06_anatomy/anat_B0_r1.json"))["anatomy"])
print("voxel anatomy == post-hoc run     :", n["anatomy_voxel"] == json.load(open("out/v06_anatomy/anatv_B0_r1.json"))["anatomy_voxel"])
r = n["reweighted_map_vs_scan"]
print("uniform-weight map == headline C  :", r["uniform_map_equals_headline"], "points", r["points"])
print("f1", n["cfg"]["f1"], "seq", n["seq"])
for s in ("outside", "global"):
    print("reweighted %-7s map %.2f per-scan %.2f voxels %d" % (s, r[s]["map"]["miou9"], r[s]["per_scan"]["miou9"], r[s]["voxels"]))
PY
echo CTL_DONE >> out/v06_mapeval/ctl_ast.txt
