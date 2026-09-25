#!/usr/bin/env bash
# opt/score_v06.sh -- v0.6 scoring.  Runs after the training queue, unattended.
#
# WHAT IT PRODUCES: every arm scored on seq 07 (the v0.4/v0.5 test sequence, which every
# design decision has been looked at) AND on seq 09 (held out in v0.6, seen for the first
# time here), at per-point and at map level.
#
# ⭐ THE TRAP THIS SCRIPT IS BUILT AROUND
# "seq 07 number minus seq 09 number" is NOT the size of the decision contamination.
# The two are different scenes -- different geometry, different class balance, different
# difficulty -- so that difference is contamination PLUS scene difficulty, and the two
# cannot be separated by looking at one arm.
#
# The zero-shot arm separates them.  No decision in this project was ever tuned against
# it (its weights are the released nuScenes checkpoint; filter E, the confidence gate and
# the voxel size were all chosen after it, and none of them changes its per-point score).
# So:
#     ZS(07) - ZS(09)                    =  scene difficulty alone
#     B0(07) - B0(09)                    =  scene difficulty + decision contamination
#     [B0(07)-B0(09)] - [ZS(07)-ZS(09)]  =  decision contamination          <- the number
# A difference-in-differences.  It is only as good as its assumption -- that the two
# scenes are equally hard for a tuned arm as for an untuned one -- and that assumption is
# stated in the report rather than hidden.
#
# ⭐ TWO FORWARD PASSES PER CHECKPOINT, NOT ONE
# The model is non-deterministic (pinned constraint 20: SerializedPooling's unstable
# torch.sort; same weights twice differ on 1-4% of points).  With 3 seeds x 2 passes the
# seed-to-seed and pass-to-pass spreads can be separated.  With one pass they cannot, and
# every seed difference would silently carry forward noise.
#
# COST: 8 models x 2 sequences x 2 passes = 32 caches (GPU, serial, ~2 h) + 32 replays
# (CPU, 4-way, ~15 min).
set -u
cd /data/livo_sem
PY=/data/miniconda3/envs/ptv3/bin/python
PYA=/data/miniconda3/envs/ags/bin/python
OUT=out/v06_score
W=weights/v06
mkdir -p $OUT $W logs
say() { echo "[$(date +'%F %T')] $*" | tee -a $OUT/score.log; }
die() { say "FAILED: $*"; exit 1; }

# ---- wait for training ----
# NOT `while is-active`: at the moment this script starts, the training unit has not been
# created yet (it is launched by prep_seq09 at the end of its own chain), so is-active
# would report inactive and this would run immediately, against nothing.  Wait for the
# queue to write a terminal line instead.
waited=0
while ! grep -qE "queue done|aborting queue|SELF-CHECK FAILED" out/v06_train/queue.log 2>/dev/null; do
    [ $waited -eq 0 ] && say "waiting for the training queue to reach a terminal state"
    sleep 300; waited=$((waited + 300))
    [ $waited -ge 259200 ] && die "training queue has not finished after 72 h"
done
if grep -qE "aborting queue|SELF-CHECK FAILED" out/v06_train/queue.log 2>/dev/null; then
    say "training queue did NOT complete normally:"
    grep -E "aborting queue|SELF-CHECK FAILED" out/v06_train/queue.log | tail -3 | tee -a $OUT/score.log
    say "scoring whatever checkpoints exist anyway, and saying so in the report"
fi
say "training queue reached a terminal state"

ARMS="armB0_s1 armB0_s2 armB0_s3 armB0_noKL_s1 armB0_noKL_s2 armB0_noKL_s3 armRprime_noKL_s1"

# ---- 1. extract students (drop the frozen KL anchor tensors) ----
say "1/4 extracting student weights"
ok_arms=""
for a in $ARMS; do
    ck=exp/sk2/$a/model/model_best.pth
    if [ ! -f "$ck" ]; then say "    $a: NO CHECKPOINT, skipping (its training job failed)"; continue; fi
    if [ ! -f "$W/${a}_student.pth" ]; then
        # --neg is the negative control: a DIFFERENT student.  Cross-model agreement must
        # sit inside same-model repeat agreement (identity), and drop clearly below it for
        # this one (difference).  Without it, "agrees 97%" proves nothing, because the
        # model agrees with itself only 96-99% anyway.
        neg=""
        [ -n "$ok_arms" ] && neg="--neg $W/$(echo $ok_arms | awk '{print $1}')_student.pth"
        $PY tools/extract_student.py --ckpt "$ck" --tag "$a" --out "$W/${a}_student.pth" \
            --report "$OUT/extract_$a.json" $neg > "logs/extract_v06_$a.log" 2>&1 \
            || { say "    $a: extraction FAILED"; continue; }
    fi
    ok_arms="$ok_arms $a"
    say "    $a  $(du -h "$W/${a}_student.pth" | cut -f1)"
done
[ -n "$ok_arms" ] || die "no student weights extracted"

# ---- 2. cache predictions: 8 models x 2 seqs x 2 passes ----
say "2/4 caching predictions (GPU, serial)"
for seq in 07 09; do
    for r in 1 2; do
        # zero-shot: no --ckpt, the released weights
        d=$OUT/pred/ZS_${seq}_r$r
        if [ ! -f "$d/.done" ]; then
            # zero-shot = the released weights: tools/cache_ptv3.py (cache_trained.py requires --ckpt)
            $PY tools/cache_ptv3.py --seq $seq --frames all --out "$d" \
                > "logs/cache_v06_ZS_${seq}_r$r.log" 2>&1 && touch "$d/.done" || say "    ZS seq$seq r$r FAILED"
        fi
        for a in $ok_arms; do
            d=$OUT/pred/${a}_${seq}_r$r
            [ -f "$d/.done" ] && continue
            $PY tools/cache_trained.py --ckpt "$W/${a}_student.pth" --seq $seq \
                --frames all --out "$d" > "logs/cache_v06_${a}_${seq}_r$r.log" 2>&1 \
                && touch "$d/.done" || say "    $a seq$seq r$r FAILED"
        done
        say "    seq$seq pass$r done ($(ls -d $OUT/pred/*_${seq}_r$r 2>/dev/null | wc -l) caches)"
    done
done

# ---- 3. map-level replay ----
say "3/4 map-level replay (CPU, 4-way)"
run() {
    tag=$1; pred=$2; traj=$3; seq_=$4
    # map level through tools/map_eval.py: replay_v05.py hard-codes the seq 07 scans and labels
    $PYA tools/map_eval.py --seq "$seq_" --pred "$pred" --pred-order raw --traj "$traj" \
        --tag "$tag" --json-out "$OUT/map_$tag.json" > "logs/replay_v06_$tag.log" 2>&1
    echo "done $tag rc=$?"
}
export -f run; export PYA OUT
for seq in 07 09; do
    case $seq in
        07) traj=out/kitti_seq07_fastlivo2_tum.txt ;;
        09) traj=out/kitti_seq09_fastlivo2_tum.txt ;;
    esac
    [ -s "$traj" ] || { say "    seq$seq: no trajectory, skipping map level"; continue; }
    for d in $OUT/pred/*_${seq}_r*; do
        [ -d "$d" ] || continue
        echo "$(basename "$d") $d $traj $seq"
    done | xargs -P 4 -L 1 bash -c 'run $0 $1 $2 $3' | tee -a $OUT/score.log
done

# ---- 4. report ----
say "4/4 report"
$PYA tools/v06_report.py > $OUT/REPORT.txt 2>&1
cat $OUT/REPORT.txt | tee -a $OUT/score.log
say "v0.6 scoring done"
