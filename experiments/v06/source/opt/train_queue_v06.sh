#!/usr/bin/env bash
# opt/train_queue_v06.sh -- v0.6 training queue.  7 jobs, STRICTLY SERIAL.
#
# WHY SERIAL: batch_size 2 measures 13.00 GB on this 4090, so two trainings do not fit
# in 24 GB.  Independently of that, nothing else may load this machine while a job runs.
#
# WHY A SELF-CHECK GATE: the whole point of v0.6 is that seq 09 is a sequence no design
# decision has seen.  A stale .pyc, a config that names the v0.4 dataset class, or a
# subclass that forgot to override SPLIT2SEQ would all silently train on seq 09 and
# quietly destroy the only clean number in the study -- after 48 h of compute.  The gate
# reads the sequence id back out of the paths actually loaded.
#
# WHY JOB 1 IS FATAL: a failure in the first job is a config error that would waste the
# remaining ~40 h.  Later failures are recorded and the queue continues, so one bad seed
# does not cost the batch.
#
#   systemd-run --user --unit=livo_v06_train --same-dir \
#       bash /data/livo_sem/opt/train_queue_v06.sh
set -u
cd /data/livo_sem
PY=/data/miniconda3/envs/ptv3/bin/python
CFGDIR=src/Pointcept_v151/configs/semantic_kitti
LOG=/data/livo_sem/out/v06_train
mkdir -p "$LOG" /data/livo_sem/exp/sk2

say() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG/queue.log"; }

# ---------------------------------------------------------------- self-check gate
say "self-check: v0.6 split"
"$PY" - > "$LOG/selfcheck.log" 2>&1 << 'PYEOF'
import sys
sys.path.insert(0, "/data/livo_sem/src")
import pointcept_ext, distil_ext, distil_ext_rprime, split_v2   # noqa: F401
from split_v2 import SPLIT2SEQ_V2, HELD_OUT, _guard

def seqs_of(paths):
    out = {}
    for p in paths:
        s = p.split("/sequences/")[1][:2]
        out[s] = out.get(s, 0) + 1
    return out

ok = True

# 1. the table
print("SPLIT2SEQ_V2 =", SPLIT2SEQ_V2)
assert SPLIT2SEQ_V2["train"] == [0, 1, 2, 4, 5, 6, 10]
assert SPLIT2SEQ_V2["val"] == [8]
assert SPLIT2SEQ_V2["test"] == [7, 9]

# 2. what actually loads, per split, read back out of the paths
for split, must, forbid in (("train", {"00","01","02","04","05","06","10"}, {"07","09"}),
                            ("val",   {"08"},                               {"07","09"}),
                            ("test",  {"07","09"},                          set())):
    # NOT test_mode=True for the test split: Pointcept's DefaultDataset then reads
    # test_cfg.voxelize, and this check supplies no test_cfg.  We only want the list
    # of paths the split loads, which get_data_list() gives in either mode.
    ds = split_v2.SemanticKITTICommon9DatasetV2(split=split)
    got = seqs_of(ds.get_data_list())
    print("%-6s %6d frames  %s" % (split, sum(got.values()), dict(sorted(got.items()))))
    if set(got) != must:
        print("  FAIL: expected sequences %s, got %s" % (sorted(must), sorted(got)))
        ok = False
    leak = set(got) & forbid
    if leak:
        print("  FAIL: held-out %s present" % sorted(leak))
        ok = False

# 3. the guard must actually fire -- a check that never fails is not a check
try:
    _guard("train", ["/x/dataset/sequences/09/velodyne/000000.bin"])
    print("FAIL: guard did not fire on a seq-09 path")
    ok = False
except AssertionError as e:
    print("guard fires as intended: %s" % str(e)[:70])

# 4. the v0.4 classes must be UNCHANGED -- v0.6 must not have leaked into them
import pointcept_ext as PX
assert PX.SemanticKITTICommon9Dataset.SPLIT2SEQ["train"] == [0, 1, 2, 4, 5, 6, 9, 10], \
    "v0.4 split was mutated; the old arms are no longer reproducible"
print("v0.4 split untouched:", PX.SemanticKITTICommon9Dataset.SPLIT2SEQ)

print("SELF-CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYEOF
if [ $? -ne 0 ] || ! grep -q "SELF-CHECK PASS" "$LOG/selfcheck.log"; then
    say "SELF-CHECK FAILED -- queue aborted, nothing trained.  See $LOG/selfcheck.log"
    tail -25 "$LOG/selfcheck.log" | tee -a "$LOG/queue.log"
    exit 1
fi
say "self-check PASS"
grep -E "^(train|val|test) " "$LOG/selfcheck.log" | tee -a "$LOG/queue.log"

# ---------------------------------------------------------------- gpu gate
wait_gpu() {
    local waited=0 used
    while :; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ "${used:-99999}" -lt 800 ] && return 0
        [ $waited -eq 0 ] && say "waiting for GPU (currently ${used} MiB used)"
        sleep 60; waited=$((waited + 60))
        if [ $waited -ge 21600 ]; then say "GPU busy 6 h -- giving up"; return 1; fi
    done
}

# ---------------------------------------------------------------- queue
JOBS="armB0_s1:arm_B0_v2_s1.py
armB0_s2:arm_B0_v2_s2.py
armB0_s3:arm_B0_v2_s3.py
armB0_noKL_s1:arm_B0_noKL_v2_s1.py
armB0_noKL_s2:arm_B0_noKL_v2_s2.py
armB0_noKL_s3:arm_B0_noKL_v2_s3.py
armRprime_noKL_s1:arm_Rprime_noKL_v2_s1.py"

n=0; total=$(echo "$JOBS" | wc -l); pass=0; fail=0
say "queue start -- $total jobs, ~6.8 h each, expect ~48 h"
echo "$JOBS" | while IFS=: read -r name cfg; do
    n=$((n + 1))
    [ -z "$name" ] && continue
    wait_gpu || exit 1
    say "job $n/$total  $name  <- $cfg"
    t0=$(date +%s)
    "$PY" tools/train_distil_v2.py \
        --config-file "$CFGDIR/$cfg" \
        --options save_path="/data/livo_sem/exp/sk2/$name" \
        > "$LOG/$name.log" 2>&1
    rc=$?
    dt=$(( ($(date +%s) - t0) / 60 ))
    best=$(grep -o "Best mIoU: [0-9.]*" "$LOG/$name.log" | tail -1)
    ck="/data/livo_sem/exp/sk2/$name/model/model_best.pth"
    if [ $rc -eq 0 ] && [ -n "$best" ] && [ -f "$ck" ]; then
        say "job $n  $name  OK   ${dt} min  $best"
        pass=$((pass + 1))
    else
        say "job $n  $name  FAILED  rc=$rc  ${dt} min  best='${best:-none}'  ckpt=$([ -f "$ck" ] && echo yes || echo NO)"
        tail -15 "$LOG/$name.log" | sed 's/^/    /' | tee -a "$LOG/queue.log"
        fail=$((fail + 1))
        if [ $n -le 1 ]; then
            say "first job failed -> aborting queue (config error would waste ~40 h)"
            exit 1
        fi
    fi
done
say "queue done"
grep -c "OK  " "$LOG/queue.log" 2>/dev/null | sed 's/^/jobs OK: /' | tee -a "$LOG/queue.log"
