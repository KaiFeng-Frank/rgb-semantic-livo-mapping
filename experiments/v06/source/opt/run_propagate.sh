#!/usr/bin/env bash
# opt/run_propagate.sh -- v0.6 map-propagated camera pseudo-labels: premise check.
# Question / instrument / gates / decision rule: out/v06_propagate/PREREG.md (written first).
# CPU only.  ONE heavy process at a time: the map_eval control runs to completion before
# propagate_check starts.  Launched with systemd-run --user; progress in logs/propagate_*.log.
set -u
cd /data/livo_sem
P=/data/miniconda3/envs/ags/bin/python
O=out/v06_propagate
LOG=logs/propagate_run.log
say() { echo "[$(date +'%F %T')] $*" | tee -a $LOG; }
[ -f $O/PREREG.md ] || { say "no PREREG.md -- stop"; exit 1; }
say "PREREG sha256 $(sha256sum $O/PREREG.md | cut -c1-16)  ($(tail -1 $O/PREREG.md))"
say "shell pid $$  cgroup $(tail -1 /proc/self/cgroup)"
echo "$$" > $O/.running_pid

# pitfall 1: no module-level name of the new tool may be assigned inside any of its functions
$P - <<'PY' >> $LOG 2>&1 || { say "AST guard FAILED"; rm -f $O/.running_pid; exit 1; }
import ast, sys
tree = ast.parse(open("tools/propagate_check.py").read())
mod = set()
for n in tree.body:
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        for al in n.names:
            mod.add((al.asname or al.name).split(".")[0])
    elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        mod.add(n.name)
    elif isinstance(n, ast.Assign):
        for t in n.targets:
            for x in ast.walk(t):
                if isinstance(x, ast.Name):
                    mod.add(x.id)
bad = []
for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
    st = {x.id for x in ast.walk(fn) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
    st |= {x.arg for x in ast.walk(fn.args) if isinstance(x, ast.arg)}
    if st & mod:
        bad.append((fn.name, sorted(st & mod)))
print("AST guard: module-level names stored inside functions:", bad or "none")
sys.exit(1 if bad else 0)
PY
say "AST guard ok"

# control: map_eval itself with NO confidence gate (keep = ok) -> the voxel key of every
# pose-valid point; propagate_check must reproduce this key set and the per-key counts.
say "control: tools/map_eval.py --conf-gate 0 --save-map"
$P tools/map_eval.py --seq 07 --pred out/v04/arms/B0_r1 --pred-order raw --conf-gate 0 \
    --tag ctl_gate0 --json-out $O/ctl_mapeval_gate0.json --save-map $O/ctl_map_gate0.npz \
    > logs/propagate_ctl_mapeval.log 2>&1 \
    || { say "control map_eval FAILED"; tail -8 logs/propagate_ctl_mapeval.log | tee -a $LOG; rm -f $O/.running_pid; exit 1; }
$P -c "import json; d=json.load(open('$O/ctl_mapeval_gate0.json')); print('control points', json.dumps(d['points']), 'voxels', d['map_voxels'], 'frames', d['frames'], 'no-pose', d['frames_no_pose'])" 2>&1 | tee -a $LOG
say "control done"

say "propagate_check"
$P tools/propagate_check.py --ctl-map $O/ctl_map_gate0.npz --json-out $O/result.json \
    --report $O/REPORT.md > logs/propagate_check.log 2>&1
rc=$?
say "propagate_check rc=$rc"
tail -14 logs/propagate_check.log | tee -a $LOG
rm -f $O/.running_pid
if [ $rc -eq 0 ]; then say "PROP_DONE"; else say "PROP_STOPPED rc=$rc"; fi
