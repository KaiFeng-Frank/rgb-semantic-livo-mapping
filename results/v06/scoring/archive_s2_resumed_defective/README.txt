out/v06_score/archive_s2_resumed_defective/ -- the RESUMED-DEFECTIVE seed 2 of arm B0_noKL.
Moved here, not deleted, by opt/rescore_s2_clean.sh on 2026-09-26 22:57:14.  Nothing in this directory is
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
