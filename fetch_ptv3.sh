#!/bin/bash
set -x
cd /data/livo_sem/src
if [ ! -d Pointcept ]; then
  git clone --depth 1 https://gh-proxy.com/https://github.com/Pointcept/Pointcept
fi
mkdir -p /data/livo_sem/weights/nuscenes-semseg-pt-v3m1-0-base/model
cd /data/livo_sem/weights/nuscenes-semseg-pt-v3m1-0-base
B=https://hf-mirror.com/Pointcept/PointTransformerV3/resolve/main/nuscenes-semseg-pt-v3m1-0-base
curl -L -f -o config.py "$B/config.py" || echo "CFG_FAIL"
curl -L -f -o model/model_best.pth "$B/model/model_best.pth" || echo "CKPT_FAIL"
ls -la . model
echo "FETCH_DONE"
