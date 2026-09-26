#!/usr/bin/env bash
# opt/rescore_s2_clean.sh -- swap the CLEAN seed 2 of arm B0_noKL into the v0.6 scoring, unattended.
#
# WHY.  armB0_noKL_s2 was resumed on 2026-09-25 through Pointcept v1.5.1's CheckpointLoader, which on
# one GPU does key = key[7:] on EVERY key: frozen_backbone.* (the frozen KL anchor = the released
# weights) was loaded into the student's backbone.*, so its epoch 10 trained the released model, not the
# seed (CRITICAL_CONSTRAINTS.md T1).  Everything opt/score_v06.sh derived from that checkpoint --
# student, extraction report, 4 caches, 4 map replays -- and the REPORT.txt that averaged them into the
# B0_noKL family is not seed 2 of B0_noKL.  opt/rerun_B0_noKL_s2.sh (unit livo_v06_rerun_s2) retrains
# the seed from scratch, no resume, into exp/sk2/armB0_noKL_s2_clean.
#
# WHAT THIS DOES.  Each step is score_v06.sh's step for this one arm: same tools, same arguments, same
# output names.  The ONLY argument that differs anywhere is extract_student.py --ckpt.
#   0. wait for RERUN_DONE in out/v06_train/queue.log (5-min polls, 12 h cap); go on only with the
#      rerun's OK line, its model_best.pth and a training log free of T1's signature.  The GPU is not
#      touched before this point.
#   1. ARCHIVE, never delete: the defective seed's 4 caches, 4 map JSONs, extraction report, student
#      weights and the REPORT.txt that averaged them -- plus their 9 logs, which steps 2-4 would
#      otherwise overwrite -- into out/v06_score/archive_s2_resumed_defective/ under their own names,
#      with README.txt and MANIFEST.txt.  Nothing is moved that is not provably the defective seed's.
#   2. extract the clean student under the SAME tag armB0_noKL_s2.  tools/v06_report.py groups runs by
#      file name (map_<arm>_<seq>_r<rep>.json, family = arm prefix "armB0_noKL_s"), so the family stays
#      seeds 1/2/3; a "_clean" tag would have joined it as a fourth seed.  Provenance of the weights:
#      out/v06_score/armB0_noKL_s2_SOURCE.txt.
#   3. cache seq 07 / 09 x passes r1 / r2 (GPU, serial, .done markers).
#   4. map-level replay of the four caches (CPU, 2-way).
#   5. prove the new artefacts came from the archived runs' invocations except the checkpoint;
#      regenerate out/v06_score/REPORT.txt; log to score.log; RESCORE_DONE.
#   6. (done first) oom_score_adj -900 on this shell, inherited by every child.
#   Any failure: "ABORTED: <why>" and RESCORE_ABORTED in out/v06_score/score.log, exit 1.
#
# CODE PIN.  The report's other 28 runs were scored 2026-09-26 13:32-15:42.  If the scoring code has
# changed since, the swapped-in seed is not comparable to them, so this script refuses (sha256 pins
# below, taken when it was written; every pinned file's mtime predates 13:32).  In that case re-score
# every arm with opt/score_v06.sh.  tools/v06_report.py is not pinned: it reads all 32 runs at once,
# so a change to it cannot make one seed incomparable to the others.
#
# SAFE TO RE-RUN.  The archive step has a completion marker.  Before it exists, only artefacts that are
# provably the defective seed's are moved (student sha256 == the out_sha256 its extraction report
# recorded for the defective checkpoint; caches and map JSONs older than the rerun's start); once it
# exists nothing is moved again, so clean results can never be archived.  Caches keep .done markers.
#
# DRY RUN.  RESCORE_DRY=1 bash opt/rescore_s2_clean.sh -- no wait, no move, no write, no GPU: every
# precondition that can be checked now is checked, every command is printed fully expanded.
#
# LAUNCH.  systemd-run --user --unit=livo_rescore_s2 --same-dir \
#              --working-directory=/data/livo_sem bash opt/rescore_s2_clean.sh
set -u
cd /data/livo_sem || exit 1
DRY=${RESCORE_DRY:-0}
PY=/data/miniconda3/envs/ptv3/bin/python
PYA=/data/miniconda3/envs/ags/bin/python
OUT=out/v06_score
W=weights/v06
ARM=armB0_noKL_s2
CK=exp/sk2/armB0_noKL_s2_clean/model/model_best.pth     # the clean from-scratch rerun
BAD_CK=exp/sk2/armB0_noKL_s2/model/model_best.pth       # the resumed-defective run (T1)
NEG=$W/armB0_s1_student.pth      # the --neg score_v06.sh gave every arm after the first
QLOG=out/v06_train/queue.log
TLOG=out/v06_train/armB0_noKL_s2_clean.log
ARCH=$OUT/archive_s2_resumed_defective
SIDE=$OUT/${ARM}_SOURCE.txt
SCORED_AT="2026-09-26 13:32:03"  # score_v06.sh began scoring the other arms (score.log)
TAGS="${ARM}_07_r1 ${ARM}_07_r2 ${ARM}_09_r1 ${ARM}_09_r2"
LOGS="logs/extract_v06_$ARM.log"
for t in $TAGS; do LOGS="$LOGS logs/cache_v06_$t.log logs/replay_v06_$t.log"; done
PINS="4c31465b8de9b4afa10552e3835db6861d552a2984b17f183bc36da708c07617  tools/extract_student.py
57b57c022be7fc776fdc20aaa80f9461fbdfd5a8b408490d48f057472e4de674  tools/cache_trained.py
0896156c9606a62d45d90bdcc83c072797619415633b7c5f5ce22fd16c6d0c6b  tools/map_eval.py
be3faa5db101a956454a8177cb0fd42d1217d9e2bfb9342d35db5465b04a1df0  src/sem_core.py
22855a13045ef86118b8acf3af278360148523575a452e5c0df9c6a988b646a6  src/label_spaces.py
5fa4b91431450773684b8bfb76a3b697c94af5187ebb117d8755bf79984f7d03  src/kitti_scan.py
ffa500dbe1729146f697d12953a9bfbfefdad7c64e72efa29e5c846fa6f1ccb7  src/kitti_calib.py
e11b7610daa9b63856417af6985419425330923cd957ffd75f464100ca8c68cd  src/seqreg.py
2632cc71f1830f5d8834f06776b20e0d7485d8ca39c35c9e7675189bd8e33efb  src/score_2d_vs_3d.py
ffd19ab64dd56323bebcaf536f422b5df4bfbe141759ccdbe31b3110f45d48a1  src/pointcept_ext.py
0e5ea9e248b8488929580464b5fc992b6f0c2fa5fb08f069a3066d39388d820e  src/distil_ext.py
fc29cbc73e2d75303445f49b14d87a585d6cefd789dd96cececd9a7732aefd1a  src/filter_e.py
284ae7bde231af309ec0343d600f6ff8938ca1a88fcf8efe8901f7a4a08ae3fe  src/random_supervise.py
834d9048c53ce1f04901541fd538198779f15758959dd5b18cf0e7782f464c94  src/ptv3_loader_verified.py
f8f39fd25643702eac9ec5b98057dc13f5fbe801d018fd8468073b653b9cd47a  src/ptv3_worker.py
1eeca94ab2f9cf2436e1e0e1a7f33333c8fa44df19b45c581929c355e4f39eff  src/ptv3_fast.py
1dd9110500d55e0bd6908746ed9ddd9a44ec266ec713463fada2c47e6741fca5  opt/replay_v03.py
2fb3249d95aecccef248aa4187445a7aeff37e50a723cf10041a3ab96887c20c  src/Pointcept_v151/configs/semantic_kitti/semseg-pt-v3m1-distil-common9.py
e76714dbd93b7eed2e436df0cc2041072d17b1225ef0fd9897283d47edb68804  weights/nuscenes-semseg-pt-v3m1-0-base/config.py"

say() {
    if [ "$DRY" = 1 ]; then echo "[dry] $*"
    else echo "[$(date +'%F %T')] rescore_s2: $*" | tee -a $OUT/score.log; fi
}
die() {
    say "ABORTED: $*"
    if [ "$DRY" != 1 ]; then
        [ -f "$ARCH/.complete" ] && [ ! -f "$OUT/REPORT.txt" ] && \
            say "    out/v06_score/REPORT.txt stays absent until a re-run of this script succeeds (it resumes); the defective-seed report is $ARCH/REPORT.txt"
        echo "RESCORE_ABORTED" >> $OUT/score.log
    fi
    exit 1
}
trap 'die "terminated by a signal"' TERM INT HUP
runlog() {            # runlog LOG CMD... : CMD > LOG 2>&1 (dry run: print it, fully expanded)
    local log=$1; shift
    if [ "$DRY" = 1 ]; then echo "[dry] RUN $* > $log 2>&1"; return 0; fi
    "$@" > "$log" 2>&1
}
check_code() {
    local bad
    bad=$(printf '%s\n' "$PINS" | sha256sum --check --quiet 2>&1) || die "scoring code changed since the $SCORED_AT scoring run, so this seed would not be comparable to the other 28 runs -- re-score every arm with opt/score_v06.sh instead: $(echo $bad)"
    bad=$(find src/Pointcept_v151/pointcept/models -name '*.py' -newermt "$SCORED_AT" | head -5)
    [ -z "$bad" ] || die "Pointcept model code changed since $SCORED_AT: $(echo $bad)"
    say "    scoring code identical to the $SCORED_AT scoring run ($(printf '%s\n' "$PINS" | grep -c .) files sha256-pinned; no Pointcept models/*.py newer)"
}
mvkeep() {            # mvkeep SRC DESTDIR : move under its own name; never overwrite, never delete
    local src=$1 dst=$2/$(basename "$1")
    [ -e "$src" ] || return 0        # absent, or already moved by an interrupted earlier attempt
    [ -e "$dst" ] && die "refusing to overwrite $dst"
    if [ "$DRY" = 1 ]; then echo "[dry] MOVE $src -> $dst"; return 0; fi
    mv -T "$src" "$dst" && [ ! -e "$src" ] && [ -e "$dst" ] || die "move $src -> $dst failed"
}
readme() {
cat <<EOF
out/v06_score/archive_s2_resumed_defective/ -- the RESUMED-DEFECTIVE seed 2 of arm B0_noKL.
Moved here, not deleted, by opt/rescore_s2_clean.sh on $(date +'%F %T').  Nothing in this directory is
seed 2 of B0_noKL, and nothing in it may be scored as such.

WHY.  Job 5 of the v0.6 training queue (armB0_noKL_s2) was killed by a global OOM on 2026-09-25 20:39
during its epoch 10 and resumed at 23:35 from its epoch-9 model_last.pth by opt/train_queue_v06_resume.sh,
through Pointcept v1.5.1's CheckpointLoader (--options resume=True weight=...).  On a single GPU that
hook does key = key[7:] on EVERY key (meant to strip "module.") and loads with strict=False.  "frozen_"
is also 7 characters, so frozen_backbone.* -- the frozen KL anchor, i.e. the released nuScenes weights in
fp16 -- was loaded into the student's backbone.*, while the student's own backbone.* / seg_head.* keys
were mangled and dropped (477 missing keys, out/v06_train/armB0_noKL_s2.resume_ep9.log).  Its epoch 10
therefore trained the released model with the epoch-9 optimizer state and the tail of the LR schedule,
not seed 2, and its save overwrote model_last.pth and model_best.pth (epoch 10, val mIoU 0.6180 =
queue.log "job 5  armB0_noKL_s2  OK   40 min").  Mechanism, and how to recognise it in a log:
CRITICAL_CONSTRAINTS.md, entry T1.

WHAT IS HERE (names unchanged; original location in brackets)
  armB0_noKL_s2_student.pth              [weights/v06/]         student extracted from
                                                                exp/sk2/armB0_noKL_s2/model/model_best.pth
  extract_armB0_noKL_s2.json             [out/v06_score/]       its extraction report
  armB0_noKL_s2_{07,09}_r{1,2}/          [out/v06_score/pred/]  per-scan prediction caches
  map_armB0_noKL_s2_{07,09}_r{1,2}.json  [out/v06_score/]       map-level replays of those caches
  REPORT.txt                             [out/v06_score/]       the v0.6 report whose B0_noKL rows
                                                                averaged this seed with seeds 1 and 3
  logs/                                  [logs/]                extract / cache / replay logs of the
                                                                above, moved so the rescore could not
                                                                overwrite them
  MANIFEST.txt                           sizes and sha256 at move time; each archived run's
                                         out-of-frustum mIoU-9
The defective checkpoint itself stays where training wrote it: exp/sk2/armB0_noKL_s2/model/.

WHAT REPLACED IT.  Seed 2 retrained from scratch, no resume (opt/rerun_B0_noKL_s2.sh, unit
livo_v06_rerun_s2) into exp/sk2/armB0_noKL_s2_clean, then extracted, cached and replayed by
opt/rescore_s2_clean.sh with score_v06.sh's own invocations under the SAME tag armB0_noKL_s2, so
tools/v06_report.py counts it as seed 2 of the B0_noKL family.  Provenance of those weights:
out/v06_score/armB0_noKL_s2_SOURCE.txt.

WHY THIS DIRECTORY CANNOT LEAK INTO THE REPORT.  tools/v06_report.py reads only
out/v06_score/map_*.json (non-recursive glob); score_v06.sh step 3 replays only out/v06_score/pred/*.
EOF
}

# ------------------------------------------------ 6. OOM priority first, so every child inherits it
if [ "$DRY" != 1 ]; then
    if echo -900 | sudo -n tee /proc/$$/oom_score_adj >/dev/null 2>&1; then
        say "start, unit shell pid $$: oom_score_adj=$(cat /proc/$$/oom_score_adj), inherited by every child"
    else
        say "start, unit shell pid $$: WARNING could not lower oom_score_adj, running unprotected"
    fi
fi
check_code

# ------------------------------------------------ 0. wait for the clean rerun
waited=0
if [ "$DRY" = 1 ]; then
    say "0/5 not waiting (dry run); RERUN_DONE lines in $QLOG now: $(grep -cx RERUN_DONE "$QLOG")"
else
    while ! grep -qx RERUN_DONE "$QLOG" 2>/dev/null; do
        [ $waited -eq 0 ] && say "0/5 waiting for RERUN_DONE in $QLOG (unit livo_v06_rerun_s2; 5-min polls, 12 h cap); GPU untouched until then"
        sleep 300; waited=$((waited + 300))
        if [ $waited -ge 43200 ] && ! grep -qx RERUN_DONE "$QLOG" 2>/dev/null; then
            die "no RERUN_DONE in $QLOG after 12 h"
        fi
    done
    say "0/5 RERUN_DONE present (waited $((waited / 60)) min)"
    check_code                      # again: the code must still be the scored version when it is used
fi
read -r n_start n_stat n_done stat < <(awk '
    index($0, "] rerun armB0_noKL_s2 (clean, no resume)") { s = NR }
    index($0, "] rerun armB0_noKL_s2_clean  OK ")         { t = NR; st = "OK" }
    index($0, "] rerun armB0_noKL_s2_clean  FAILED")      { t = NR; st = "FAILED" }
    $0 == "RERUN_DONE"                                    { d = NR }
    END { print s + 0, t + 0, d + 0, (st == "" ? "NONE" : st) }' "$QLOG")
[ "$n_start" -gt 0 ] || die "no 'rerun armB0_noKL_s2 (clean, no resume)' start line in $QLOG"
start_str=$(sed -n "${n_start}p" "$QLOG" | sed -n 's/^\[\([0-9-]* [0-9:]*\)\].*/\1/p')
[ -n "$start_str" ] || die "no timestamp on the rerun start line $n_start of $QLOG"   # date -d "" = midnight
T0=$(date -d "$start_str" +%s) || die "cannot parse the rerun start time '$start_str'"
ok_line=""; [ "$n_stat" -gt 0 ] && ok_line=$(sed -n "${n_stat}p" "$QLOG")
best=$(printf '%s\n' "$ok_line" | sed -n 's/.*Best mIoU: \([0-9.]*\).*/\1/p')
say "    queue.log: rerun started $start_str (line $n_start); status line $n_stat: ${ok_line:-none yet}; last RERUN_DONE line $n_done"
if [ "$DRY" != 1 ]; then
    [ "$n_done" -gt "$n_start" ] || die "the latest rerun (queue.log line $n_start) has not written RERUN_DONE"
    [ "$stat" = OK ] && [ "$n_stat" -gt "$n_start" ] && [ "$n_stat" -lt "$n_done" ] \
        || die "no OK line for the latest rerun (status $stat, line $n_stat) -- nothing clean to score"
    [ -n "$best" ] || die "the OK line carries no 'Best mIoU'"
fi
[ -s "$CK" ] || die "$CK is missing"
grep -q "Missing keys" "$TLOG" && die "$TLOG contains 'Missing keys' -- the CRITICAL_CONSTRAINTS.md T1 signature"
grep -q "No weight found at: None" "$TLOG" || die "$TLOG lacks 'No weight found at: None' -- not a from-scratch run"
say "    $CK present; $TLOG: 'No weight found at: None', no 'Missing keys' (from scratch, T1 signature absent)"

# ------------------------------------------------ 1. archive the resumed-defective seed
say "1/5 archiving the resumed-defective seed -> $ARCH"
if [ -f "$ARCH/.complete" ]; then
    say "    archive already complete since $(cat "$ARCH/.complete") -- nothing moved (this is a re-run)"
else
    $PYA - "$OUT" "$ARCH" "$ARM" "$BAD_CK" "$W" "$T0" $LOGS <<'PYEOF' || die "provenance check failed -- nothing was moved"
import hashlib, json, os, sys, time
OUT, ARCH, ARM, BAD_CK, W, T0 = sys.argv[1:7]
LOGS = sys.argv[7:]
T0 = float(T0)
bad = []


def where(src, sub=""):
    dst = os.path.join(ARCH, sub, os.path.basename(src))
    a, b = os.path.exists(src), os.path.exists(dst)
    if a and b:
        bad.append("present in both places: %s and %s" % (src, dst))
    return src if a else (dst if b else None)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def ts(p):
    return time.strftime("%F %T", time.localtime(os.path.getmtime(p)))


print("    provenance (rerun started %s; every cache and map JSON to be moved must predate it):"
      % time.strftime("%F %T", time.localtime(T0)))
rep_json = where("%s/extract_%s.json" % (OUT, ARM))
student = where("%s/%s_student.pth" % (W, ARM))
if not (rep_json and student):
    bad.append("extraction report or student missing (%s, %s)" % (rep_json, student))
else:
    e = json.load(open(rep_json))
    if e.get("ckpt") != BAD_CK or e.get("tag") != ARM:
        bad.append("%s records ckpt %r tag %r, not the defective %r / %r"
                   % (rep_json, e.get("ckpt"), e.get("tag"), BAD_CK, ARM))
    s = sha(student)
    if s != e.get("out_sha256"):
        bad.append("%s sha256 %s is not the out_sha256 %s recorded in %s"
                   % (student, s[:16], str(e.get("out_sha256"))[:16], rep_json))
    print("      %s sha256 %s == out_sha256 in %s (extracted %s from %s, epoch %s, val %s)"
          % (student, s[:16], os.path.basename(rep_json), e.get("time"), e.get("ckpt"),
             e.get("source", {}).get("epoch"), e.get("source", {}).get("best_metric_value")))
for seq in ("07", "09"):
    for r in ("1", "2"):
        t = "%s_%s_r%s" % (ARM, seq, r)
        d = where("%s/pred/%s" % (OUT, t))
        j = where("%s/map_%s.json" % (OUT, t))
        if not (d and j):
            bad.append("%s: cache %s / map json %s missing" % (t, d, j))
            continue
        try:
            m = json.load(open(os.path.join(d, "meta.json")))
        except Exception as err:
            m = {}
            bad.append("%s/meta.json unreadable: %s" % (d, err))
        done = os.path.join(d, ".done")
        if m.get("ckpt") != "%s/%s_student.pth" % (W, ARM) or m.get("seq") != seq:
            bad.append("%s: meta ckpt %r seq %r" % (d, m.get("ckpt"), m.get("seq")))
        if not os.path.exists(done) or os.path.getmtime(done) >= T0:
            bad.append("%s/.done missing or not older than the rerun" % d)
        mj = json.load(open(j))
        if mj.get("tag") != t or mj.get("pred") != "%s/pred/%s" % (OUT, t):
            bad.append("%s: tag %r pred %r" % (j, mj.get("tag"), mj.get("pred")))
        if os.path.getmtime(j) >= T0:
            bad.append("%s is not older than the rerun" % j)
        n = len([f for f in os.listdir(d) if f.endswith(".npz")])
        print("      %-22s cache %4d npz, .done %s | map json %s, pred=%s"
              % (t, n, ts(done) if os.path.exists(done) else "-", ts(j), mj.get("pred")))
rep = where("%s/REPORT.txt" % OUT)
print("      REPORT.txt: %s" % (("%s (%s)" % (rep, ts(rep))) if rep else "absent -- nothing to archive"))
print("      logs: %d of %d present" % (sum(1 for l in LOGS if where(l, "logs")), len(LOGS)))
for b in bad:
    print("    PROVENANCE FAIL: " + b)
sys.exit(1 if bad else 0)
PYEOF
    if [ "$DRY" = 1 ]; then
        echo "[dry] WRITE $ARCH/README.txt:"; readme | sed 's/^/[dry]   | /'
    else
        mkdir -p "$ARCH/logs" || die "cannot create $ARCH/logs"
        [ -f "$ARCH/README.txt" ] || readme > "$ARCH/README.txt" || die "cannot write $ARCH/README.txt"
    fi
    for t in $TAGS; do mvkeep "$OUT/pred/$t" "$ARCH"; done
    for t in $TAGS; do mvkeep "$OUT/map_$t.json" "$ARCH"; done
    mvkeep "$OUT/extract_$ARM.json" "$ARCH"
    mvkeep "$W/${ARM}_student.pth" "$ARCH"
    mvkeep "$OUT/REPORT.txt" "$ARCH"
    for l in $LOGS; do mvkeep "$l" "$ARCH/logs"; done
    if [ "$DRY" != 1 ]; then
        $PYA - "$ARCH" > "$ARCH/MANIFEST.txt" <<'PYEOF' || die "cannot write $ARCH/MANIFEST.txt"
import hashlib, json, os, sys, time
ARCH = sys.argv[1]
M = "miou9_abstain_excluded"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


out = ["# %s -- written %s by opt/rescore_s2_clean.sh right after the move"
       % (ARCH, time.strftime("%F %T")),
       "# files: bytes + sha256 | cache dirs: bytes, file count, meta.json"]
for n in sorted(os.listdir(ARCH)):
    p = os.path.join(ARCH, n)
    if n in ("MANIFEST.txt", ".complete", "README.txt"):
        continue
    if n == "logs":
        for l in sorted(os.listdir(p)):
            q = os.path.join(p, l)
            out.append("logs/%-45s %11d  %s" % (l, os.path.getsize(q), sha(q)))
    elif os.path.isdir(p):
        fs = [os.path.join(p, f) for f in os.listdir(p)]
        m = json.load(open(os.path.join(p, "meta.json")))
        out.append("%-50s %11d  %d files; meta ckpt=%s seq=%s n_frames=%s"
                   % (n + "/", sum(os.path.getsize(f) for f in fs), len(fs),
                      m.get("ckpt"), m.get("seq"), m.get("n_frames")))
    else:
        out.append("%-50s %11d  %s" % (n, os.path.getsize(p), sha(p)))
out += ["", "# out-of-frustum mIoU-9 of the archived (defective) runs, the fields tools/v06_report.py reads"]
for n in sorted(os.listdir(ARCH)):
    if n.startswith("map_") and n.endswith(".json"):
        d = json.load(open(os.path.join(ARCH, n)))
        out.append("%-42s offline_all %6.2f   map_all_lookup %6.2f"
                   % (n, d["offline_all"]["outside"][M], d["map_all_lookup"]["outside"][M]))
print("\n".join(out))
PYEOF
        date +'%F %T' > "$ARCH/.complete"
        say "    moved 4 caches, 4 map JSONs, the extraction report, the student, REPORT.txt and $(ls "$ARCH/logs" | wc -l) logs; README.txt + MANIFEST.txt written"
    fi
fi

# ------------------------------------------------ 2. extract the clean student (same tag)
say "2/5 extracting the clean student: $CK -> $W/${ARM}_student.pth (tag $ARM)"
[ -s "$NEG" ] || die "negative control $NEG is missing"
if [ "$DRY" != 1 ]; then
    say "    GPU now: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | tr -d '\n'); compute apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | tr '\n' ' ')"
    ck_sha=$(sha256sum "$CK" | cut -d' ' -f1)
fi
# always run (cheap, ~1 min): a failed earlier extraction may have left a student behind
runlog "logs/extract_v06_$ARM.log" $PY tools/extract_student.py --ckpt "$CK" --tag "$ARM" \
    --out "$W/${ARM}_student.pth" --report "$OUT/extract_$ARM.json" --neg "$NEG" \
    || die "extraction FAILED (logs/extract_v06_$ARM.log)"
if [ "$DRY" = 1 ]; then
    echo "[dry] then: check $OUT/extract_$ARM.json (PASS, ckpt/tag/out/neg, source sha256 == sha256sum $CK,"
    echo "[dry]       best_metric_value == the OK line's Best mIoU, 488/488/0 tensors, student sha256) -> WRITE $SIDE"
else
    $PYA - "$OUT/extract_$ARM.json" "$CK" "$ck_sha" "$ARM" "$W/${ARM}_student.pth" "$NEG" "$best" \
        "$ok_line" "$TLOG" "$SIDE" "$ARCH/extract_$ARM.json" <<'PYEOF' || die "the clean extraction does not check out"
import hashlib, json, os, sys, time
REP, CK, CK_SHA, ARM, STUDENT, NEG, BEST, OKLINE, TLOG, SIDE, OLD = sys.argv[1:12]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


e = json.load(open(REP))
s = e.get("source", {})
bad = []


def need(ok, msg):
    if not ok:
        bad.append(msg)


need(e.get("PASS") is True and not e.get("fails"), "extraction verdict: %s" % e.get("fails"))
need(e.get("ckpt") == CK, "ckpt %r" % e.get("ckpt"))
need(e.get("tag") == ARM, "tag %r" % e.get("tag"))
need(e.get("out") == STUDENT, "out %r" % e.get("out"))
need(e.get("neg") == NEG, "neg %r" % e.get("neg"))
need(s.get("sha256") == CK_SHA, "source sha256 %s != sha256sum %s" % (s.get("sha256"), CK_SHA))
bmv = s.get("best_metric_value")
need(bmv is not None and "%.4f" % bmv == BEST,
     "checkpoint best_metric_value %r is not the OK line's Best mIoU %s" % (bmv, BEST))
need((s.get("n_student"), s.get("n_anchor"), s.get("n_other")) == (488, 488, 0),
     "tensor partition student/anchor/other %s/%s/%s" % (s.get("n_student"), s.get("n_anchor"), s.get("n_other")))
st_sha = sha(STUDENT)
need(st_sha == e.get("out_sha256"), "student sha256 %s != out_sha256 %s" % (st_sha, e.get("out_sha256")))
old = json.load(open(OLD))
need(old["source"]["sha256"] != CK_SHA, "clean checkpoint is byte-identical to the defective one")
for b in bad:
    print("    EXTRACT CHECK FAIL: " + b)
if bad:
    sys.exit(1)
tools = ["tools/extract_student.py", "tools/cache_trained.py", "tools/map_eval.py", "tools/v06_report.py"]
L = ["# %s -- where the weights scored under tag %s come from" % (SIDE, ARM),
     "# written %s by opt/rescore_s2_clean.sh" % time.strftime("%F %T"),
     "tag                 %s  (kept, so tools/v06_report.py counts it as seed 2 of the B0_noKL family)" % ARM,
     "source_checkpoint   %s" % CK,
     "source_sha256       %s" % CK_SHA,
     "source_bytes        %s" % s.get("bytes"),
     "source_epoch        %s  (as recorded in the checkpoint)" % s.get("epoch"),
     "source_best_val     %r  (seq 08 val mIoU)" % bmv,
     "queue_line          %s" % OKLINE,
     "training            opt/rerun_B0_noKL_s2.sh (unit livo_v06_rerun_s2): from scratch, no resume;",
     "                    log %s: 'No weight found at: None', no 'Missing keys'" % TLOG,
     "student             %s  sha256 %s  (%d bytes; backbone.* + seg_head.* = 488 tensors, frozen_* dropped)"
     % (STUDENT, st_sha, os.path.getsize(STUDENT)),
     "extraction_report   %s  PASS (S1-S4), extracted %s, negative control %s" % (REP, e.get("time"), NEG),
     "replaces            %s  sha256 %s" % (old.get("ckpt"), old["source"]["sha256"]),
     "                    resumed through Pointcept v1.5.1's CheckpointLoader, frozen_backbone.* loaded into",
     "                    backbone.* (CRITICAL_CONSTRAINTS.md T1); its artefacts: out/v06_score/archive_s2_resumed_defective/",
     "scoring_code        identical to the 2026-09-26 13:32 scoring of the other arms (sha256-pinned in opt/rescore_s2_clean.sh):"]
L += ["  %s  %s" % (sha(t), t) for t in tools]
open(SIDE, "w").write("\n".join(L) + "\n")
print("    %s: PASS; source sha256 %s = sha256sum; best %.4f = OK line; student sha256 %s -> %s"
      % (REP, CK_SHA[:16], bmv, st_sha[:16], SIDE))
PYEOF
fi

# ------------------------------------------------ 3. caches (GPU, serial) -- score_v06.sh step 2 for one arm
say "3/5 caching predictions (GPU, serial)"
for seq in 07 09; do
    for r in 1 2; do
        d=$OUT/pred/${ARM}_${seq}_r$r
        if [ "$DRY" != 1 ] && [ -f "$d/.done" ]; then
            say "    ${ARM}_${seq}_r$r: .done present (earlier attempt of this script, after the archive) -- kept"
            continue
        fi
        for attempt in 1 2; do
            runlog "logs/cache_v06_${ARM}_${seq}_r$r.log" $PY tools/cache_trained.py --ckpt "$W/${ARM}_student.pth" --seq $seq \
                --frames all --out "$d" && break
            [ $attempt -eq 1 ] || die "cache ${ARM}_${seq}_r$r FAILED twice (logs/cache_v06_${ARM}_${seq}_r$r*.log)"
            cp -f "logs/cache_v06_${ARM}_${seq}_r$r.log" "logs/cache_v06_${ARM}_${seq}_r$r.attempt1.log"
            say "    ${ARM}_${seq}_r$r failed once, retrying (first log kept as *.attempt1.log)"
        done
        if [ "$DRY" != 1 ]; then
            touch "$d/.done"
            say "    ${ARM}_${seq}_r$r cached ($(ls "$d" | grep -c '\.npz$') frames)"
        fi
    done
done

# ------------------------------------------------ 4. map-level replay (CPU) -- score_v06.sh step 3 for one arm
say "4/5 map-level replay (CPU, 2-way)"
run() {
    tag=$1; pred=$2; traj=$3; seq_=$4
    set -- $PYA tools/map_eval.py --seq "$seq_" --pred "$pred" --pred-order raw --traj "$traj" \
        --tag "$tag" --json-out "$OUT/map_$tag.json"
    if [ "$DRY" = 1 ]; then echo "[dry] RUN $* > logs/replay_v06_$tag.log 2>&1"; return 0; fi
    "$@" > "logs/replay_v06_$tag.log" 2>&1
    echo "done $tag rc=$?"
}
export -f run; export PYA OUT DRY
replayed=""
for seq in 07 09; do
    case $seq in
        07) traj=out/kitti_seq07_fastlivo2_tum.txt ;;
        09) traj=out/kitti_seq09_fastlivo2_tum.txt ;;
    esac
    [ -s "$traj" ] || die "seq$seq: no trajectory $traj"
    for r in 1 2; do
        d=$OUT/pred/${ARM}_${seq}_r$r
        [ "$DRY" = 1 ] || [ -f "$d/.done" ] || die "$d has no .done"
        # after step 1 this file can only be an earlier attempt's own output
        [ "$DRY" = 1 ] || rm -f "$OUT/map_$(basename "$d").json"
    done
    P=2
    avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
    if [ "$avail" -lt 30 ]; then
        P=1; say "    seq$seq: MemAvailable ${avail} GiB < 30 (one seq-09 replay peaks ~12 GiB) -> 1-way"
    fi
    res=$(for r in 1 2; do
        d=$OUT/pred/${ARM}_${seq}_r$r
        echo "$(basename "$d") $d $traj $seq"
    done | xargs -P $P -L 1 bash -c 'run $0 $1 $2 $3')
    if [ "$DRY" = 1 ]; then echo "$res"; else echo "$res" | tee -a $OUT/score.log; fi
    replayed="$replayed
$res"
done
if [ "$DRY" != 1 ]; then
    for t in $TAGS; do
        printf '%s\n' "$replayed" | grep -qx "done $t rc=0" || die "replay $t did not finish with rc=0 (logs/replay_v06_$t.log)"
        [ -s "$OUT/map_$t.json" ] || die "replay $t wrote no $OUT/map_$t.json"
    done
fi

# ------------------------------------------------ 5. prove the swap, then the report
say "5/5 checking the new artefacts against the archived ones, then the report"
mode=full; [ "$DRY" = 1 ] && mode=family
$PYA - "$OUT" "$ARCH" "$ARM" "$CK" "$BAD_CK" "$mode" <<'PYEOF' || die "post-check failed; REPORT.txt not regenerated"
import glob, json, os, re, sys
OUT, ARCH, ARM, CK, BAD_CK, MODE = sys.argv[1:7]
TAG_RE = re.compile(r"^map_(?P<arm>.+)_(?P<seq>\d{2})_r(?P<rep>\d+)\.json$")      # = tools/v06_report.py
M = "miou9_abstain_excluded"
bad = []
# (1) the B0_noKL family exactly as tools/v06_report.py's load_all() + fam() will build it
got = {}
for f in sorted(glob.glob(os.path.join(OUT, "map_*.json"))):
    m = TAG_RE.match(os.path.basename(f))
    if m and m["arm"].startswith("armB0_noKL_s") and "noKL" not in m["arm"][len("armB0_noKL_s"):]:
        got.setdefault(m["arm"], []).append((m["seq"], m["rep"]))
want = {a: [(s, r) for s in ("07", "09") for r in ("1", "2")]
        for a in ("armB0_noKL_s1", "armB0_noKL_s2", "armB0_noKL_s3")}
got = {a: sorted(v) for a, v in got.items()}
if got != want:
    bad.append("B0_noKL family as tools/v06_report.py groups it: %s" % got)
print("    B0_noKL family as tools/v06_report.py groups it: %s"
      % ", ".join("%s x%d" % (a, len(v)) for a, v in sorted(got.items())))
if MODE == "full":
    # (2) same invocations as the archived runs; the only difference allowed is the checkpoint
    new = json.load(open("%s/extract_%s.json" % (OUT, ARM)))
    old = json.load(open("%s/extract_%s.json" % (ARCH, ARM)))
    for k in ("tag", "out", "neg", "frames"):
        if new.get(k) != old.get(k):
            bad.append("extract %s: new %r archived %r" % (k, new.get(k), old.get(k)))
    if (new.get("ckpt"), old.get("ckpt")) != (CK, BAD_CK):
        bad.append("extract ckpt: new %r archived %r" % (new.get("ckpt"), old.get("ckpt")))
    if new["out_sha256"] == old["out_sha256"]:
        bad.append("the new student is byte-identical to the defective one")
    print("    extract: tag/out/neg/frames identical to the archived run; ckpt %s (archived: %s)"
          % (new.get("ckpt"), old.get("ckpt")))
    for seq in ("07", "09"):
        for r in ("1", "2"):
            t = "%s_%s_r%s" % (ARM, seq, r)
            nd, od = "%s/pred/%s" % (OUT, t), "%s/%s" % (ARCH, t)
            nm, om = json.load(open(nd + "/meta.json")), json.load(open(od + "/meta.json"))
            for k in ("ckpt", "seq", "n_frames", "space", "intensity_scale", "grid", "shuffle_orders", "tta"):
                if nm.get(k) != om.get(k):
                    bad.append("%s cache meta %s: new %r archived %r" % (t, k, nm.get(k), om.get(k)))
            nz = len(glob.glob(nd + "/f*.npz"))
            if nz != nm.get("n_frames") or not os.path.exists(nd + "/.done"):
                bad.append("%s: %d npz for n_frames %s, .done %s" % (t, nz, nm.get("n_frames"), os.path.exists(nd + "/.done")))
            nj = json.load(open("%s/map_%s.json" % (OUT, t)))
            oj = json.load(open("%s/map_%s.json" % (ARCH, t)))
            for k in ("tag", "pred", "pred_order", "traj", "seq", "cfg", "frames", "frames_no_pose"):
                if nj.get(k) != oj.get(k):
                    bad.append("%s map json %s: new %r archived %r" % (t, k, nj.get(k), oj.get(k)))
            print("    %-22s cache %4d frames; replay tag/pred/traj/cfg as archived; out-of-frustum mIoU-9"
                  " offline %.2f map %.2f (archived defective run: %.2f / %.2f)"
                  % (t, nz, nj["offline_all"]["outside"][M], nj["map_all_lookup"]["outside"][M],
                     oj["offline_all"]["outside"][M], oj["map_all_lookup"]["outside"][M]))
for b in bad:
    print("    POST-CHECK FAIL: " + b)
sys.exit(1 if bad else 0)
PYEOF
if [ "$DRY" = 1 ]; then
    echo "[dry] RUN $PYA tools/v06_report.py > $OUT/REPORT.txt.new 2>&1; mv -f $OUT/REPORT.txt.new $OUT/REPORT.txt"
    echo "[dry] then: one summary line + RESCORE_DONE appended to $OUT/score.log"
    exit 0
fi
$PYA tools/v06_report.py > $OUT/REPORT.txt.new 2>&1 || die "tools/v06_report.py failed (its output: $OUT/REPORT.txt.new)"
mv -f $OUT/REPORT.txt.new $OUT/REPORT.txt || die "cannot put $OUT/REPORT.txt in place"
cat $OUT/REPORT.txt
say "REPORT.txt regenerated with the CLEAN seed 2 of B0_noKL ($CK, sha256 ${ck_sha:0:16}; provenance $SIDE; defective seed in $ARCH): $(head -1 $OUT/REPORT.txt) | B0_noKL row: $(grep -m1 '^B0_noKL ' $OUT/REPORT.txt | tr -s ' ')"
echo "RESCORE_DONE" >> $OUT/score.log
