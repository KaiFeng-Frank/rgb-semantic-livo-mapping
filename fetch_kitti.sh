#!/bin/bash
set -u
D=/data/livo_sem/data; cd $D
B="https://s3.eu-central-1.amazonaws.com/avg-kitti"
get(){ n="$1"; u="$2"
  echo "=== [$(date +%H:%M:%S)] 下载 $n"
  wget -c --progress=dot:giga -T30 -t5 -O "$n" "$u" 2>&1 | grep -E "已保存|saved|%" | tail -3
  echo "    -> $(ls -lh $n 2>/dev/null | awk '{print $5}')"
}
get 2011_09_30_calib.zip            "$B/raw_data/2011_09_30_calib.zip"
get data_odometry_calib.zip         "$B/data_odometry_calib.zip"
get data_odometry_labels.zip        "http://www.semantic-kitti.org/assets/data_odometry_labels.zip"
get 2011_09_30_drive_0016_sync.zip  "$B/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip"
get 2011_09_30_drive_0027_sync.zip  "$B/raw_data/2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip"

echo "=== [$(date +%H:%M:%S)] 解压 ==="
for z in 2011_09_30_calib.zip 2011_09_30_drive_0016_sync.zip 2011_09_30_drive_0027_sync.zip; do
  echo "--- unzip $z"; unzip -q -o "$z" -d $D/raw/ && echo "    ok"
done
mkdir -p $D/odometry
unzip -q -o data_odometry_calib.zip  -d $D/odometry/ && echo "--- odometry calib ok"
unzip -q -o data_odometry_labels.zip -d $D/odometry/ && echo "--- semantickitti labels ok"

echo "=== [$(date +%H:%M:%S)] 清点 ==="
echo "raw drives:"; ls $D/raw/2011_09_30/ 2>/dev/null
echo "seq07 velodyne 帧数: $(ls $D/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/ 2>/dev/null | wc -l)"
echo "seq07 cam2(彩色) 帧数: $(ls $D/raw/2011_09_30/2011_09_30_drive_0027_sync/image_02/data/ 2>/dev/null | wc -l)"
echo "seq07 oxts(IMU) 帧数:  $(ls $D/raw/2011_09_30/2011_09_30_drive_0027_sync/oxts/data/ 2>/dev/null | wc -l)"
echo "seq04 velodyne 帧数: $(ls $D/raw/2011_09_30/2011_09_30_drive_0016_sync/velodyne_points/data/ 2>/dev/null | wc -l)"
echo "semantic labels 序列: $(ls $D/odometry/dataset/sequences/ 2>/dev/null | tr '\n' ' ')"
echo "seq07 label 帧数: $(ls $D/odometry/dataset/sequences/07/labels/ 2>/dev/null | wc -l)"
echo "seq04 label 帧数: $(ls $D/odometry/dataset/sequences/04/labels/ 2>/dev/null | wc -l)"
df -h /data | tail -1
echo "=== ALL DONE ==="
