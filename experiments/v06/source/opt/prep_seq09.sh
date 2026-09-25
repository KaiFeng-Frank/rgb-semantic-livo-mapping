#!/usr/bin/env bash
# opt/prep_seq09.sh -- build the seq 09 test rig, then hand off to the training queue.
#
# WHY: v0.6 holds seq 09 out of training so one test sequence exists that no design
# decision of v0.4/v0.5 has ever seen.  Per-point scoring on it needs only points and
# labels, which are already on disk.  MAP-LEVEL scoring -- the v0.5 finding that has to
# be reproduced on clean data -- needs a bag and a FAST-LIVO2 trajectory, which do not
# exist yet.
#
# WHAT IS ALREADY HERE (checked 2026-09-24, do not re-download):
#   data/raw/2011_09_30/2011_09_30_drive_0033_sync   1594 frames, velodyne+cam2+cam3+oxts
#   data/odometry/dataset/sequences/09/{labels,poses.txt,calib.txt}
#   data/raw/2011_09_30/calib_*.txt                  same calibration day as seq 07
#
# WHAT IS MISSING: the 100 Hz IMU.  The sync drive's oxts is 10 Hz (camera-synchronised);
# FAST-LIVO2 needs the extract drive's 100 Hz stream.  KITTI only ships it inside the
# full 9 GB extract archive, so the archive is fetched, the oxts subtree extracted, and
# the archive deleted -- the same path fetch_oxts100.sh took for seq 04 / seq 07.
#
# FRAME COUNT: KITTI odometry seq 09 is drive 0033 frames 000000-001590, so --end 1591,
# exactly as seq 07 is drive 0027 frames 000000-001100 with --end 1101 in build_bags.sh.
#
# This script runs ALONE: FAST-LIVO2 is a real-time run and anything else on the machine
# distorts it.
set -u
cd /data/livo_sem
D=/data/livo_sem/data
OUT=out/v06_prep
mkdir -p "$OUT" logs
say() { echo "[$(date +'%F %T')] $*" | tee -a $OUT/prep.log; }
die() { say "FAILED: $*"; exit 1; }

# ---- [removed from the public copy: a wait on an unrelated job that shared the host] ----
say "machine clear"

# ---- 1. 100 Hz oxts ----
OX=$D/extract/2011_09_30/2011_09_30_drive_0033_extract/oxts/data
if [ -d "$OX" ] && [ "$(ls "$OX" 2>/dev/null | wc -l)" -gt 10000 ]; then
    say "1/4 100 Hz oxts already present ($(ls "$OX" | wc -l) files), skipping download"
else
    Z=$D/2011_09_30_drive_0033_extract.zip
    say "1/4 fetching 0033_extract (9.1 GB, for the oxts subtree only)"
    wget -c -q -T30 -t5 -O "$Z" \
      "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0033/2011_09_30_drive_0033_extract.zip" \
      || die "download"
    say "    got $(du -h "$Z" | cut -f1); extracting oxts"
    unzip -q -o "$Z" '*/oxts/*' -d "$D/extract/" || die "unzip"
    rm -f "$Z"
    n=$(ls "$OX" 2>/dev/null | wc -l)
    [ "$n" -gt 10000 ] || die "only $n oxts files after extraction (expected ~16000)"
    say "    $n oxts samples, archive deleted"
fi

# ---- 2. bags ----
if [ -f bags/kitti_seq09_us/metadata.yaml ]; then
    say "2/4 bags already built, skipping"
else
    say "2/4 building bags (--end 1591)"
    # ROS's setup.bash reads AMENT_TRACE_SETUP_FILES without defining it, so `set -u`
    # kills the shell the moment it is sourced.  run_fastlivo2_kitti.sh opens with
    # `set +u` for exactly this reason.
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
    for suf in "" "_us"; do
        extra=""; [ -n "$suf" ] && extra="--time-unit us"
        rm -rf "bags/kitti_seq09$suf"
        /usr/bin/python3 src/kitti_to_ros2bag.py \
            --drive $D/raw/2011_09_30/2011_09_30_drive_0033_sync \
            --oxts100 $D/extract/2011_09_30/2011_09_30_drive_0033_extract/oxts \
            --out "bags/kitti_seq09$suf" --end 1591 $extra \
            > "logs/bag_seq09$suf.log" 2>&1 || die "bag build kitti_seq09$suf"
        say "    bags/kitti_seq09$suf  $(du -sh bags/kitti_seq09$suf | cut -f1)"
    done
fi

# ---- 3. FAST-LIVO2 trajectory ----
if [ -s out/kitti_seq09_fastlivo2_tum.txt ]; then
    say "3/4 trajectory already present, skipping"
else
    say "3/4 running FAST-LIVO2 on seq 09 (real-time run, machine must stay idle)"
    bash run_fastlivo2_kitti.sh kitti_seq09 /data/livo_sem/bags/kitti_seq09_us 0.5 \
        > logs/fastlivo2_seq09_driver.log 2>&1 || die "FAST-LIVO2 run"
    [ -s out/kitti_seq09_fastlivo2_tum.txt ] || die "no trajectory written"
fi
n=$(wc -l < out/kitti_seq09_fastlivo2_tum.txt)
say "    trajectory: $n poses (seq 09 has 1591 frames)"
[ "$n" -ge 1400 ] || die "trajectory has only $n poses -- FAST-LIVO2 lost tracking"

# ---- 4. sanity: the rig must look like seq 07's ----
say "4/4 rig check"
t0=$(head -1 out/kitti_seq09_fastlivo2_tum.txt | cut -d' ' -f1)
t1=$(tail -1 out/kitti_seq09_fastlivo2_tum.txt | cut -d' ' -f1)
say "    t in [$t0, $t1]   labels=$(ls $D/odometry/dataset/sequences/09/labels | wc -l)   gt_poses=$(wc -l < $D/odometry/dataset/sequences/09/poses.txt)"

say "seq 09 rig ready -- handing off to the training queue"
systemd-run --user --unit=livo_v06_train --same-dir \
    --working-directory=/data/livo_sem \
    bash /data/livo_sem/opt/train_queue_v06.sh 2>&1 | tail -1 | tee -a $OUT/prep.log
