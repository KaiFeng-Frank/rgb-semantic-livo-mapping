#!/bin/bash
set -u
D=/data/livo_sem/data
cd $D
B="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
for d in 0016 0027; do
  Z=2011_09_30_drive_${d}_extract.zip
  echo "=== [$(date +%H:%M:%S)] download $Z"
  wget -c -q -T30 -t5 -O "$Z" "$B/2011_09_30_drive_${d}/$Z"
  echo "    got $(ls -lh $Z | awk '{print $5}')"
  echo "=== [$(date +%H:%M:%S)] extract oxts only from $Z"
  unzip -q -o "$Z" '*/oxts/*' -d $D/extract/ && echo "    unzip ok"
  rm -f "$Z"
  echo "    removed zip"
done
echo "=== oxts100 counts ==="
for d in 0016 0027; do
  echo "$d: $(ls $D/extract/2011_09_30/2011_09_30_drive_${d}_extract/oxts/data 2>/dev/null | wc -l) files"
done
df -h /data | tail -1
echo "=== OXTS100 DONE ==="
