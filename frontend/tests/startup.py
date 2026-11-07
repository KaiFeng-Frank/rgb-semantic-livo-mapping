"""Verify the actual camera service and both estimator startup modes without data."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from frontend.replay_smoke import terminate

REPO = Path(__file__).resolve().parents[2]


def check(workspace, out):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ROS_DOMAIN_ID='174', RCUTILS_LOGGING_BUFFERED_STREAM='0')
    exe = workspace/'install/fast_livo/lib/fast_livo/fastlivo_mapping'
    for name in ('geometry','semantic','enhanced'):
        enabled=name!='geometry'
        extra=[]
        if name=='enhanced':
            for key in ('dynamics','visual_mask','likelihood','adaptive_boundary_noise'):
                extra+=['-p','semantic.'+key+':=true']
        with (out/(name+'.camera.log')).open('w') as camera_log, (out/(name+'.log')).open('w') as log:
            camera = subprocess.Popen([sys.executable, str(REPO/'frontend/parameter_blackboard.py'),
                '--ros-args', '--params-file', str(REPO/'config/camera_kitti_cam2.yaml')],
                env=env, stdout=camera_log, stderr=subprocess.STDOUT, start_new_session=True)
            node = None
            try:
                time.sleep(1)
                node = subprocess.Popen([str(exe), '--ros-args', '--params-file',
                    str(REPO/'config/kitti_velodyne64.yaml'), '-p', 'semantic.enabled:='+str(enabled).lower(),*extra],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                for _ in range(150):
                    if camera.poll() is not None or node.poll() is not None:
                        raise RuntimeError(name+' startup failed; see '+str(out))
                    text = (out/(name+'.log')).read_text()
                    if 'intrinsic: 707.091200, 707.091200, 601.887300, 183.110400' in text:
                        time.sleep(1)
                        if node.poll() is not None: raise RuntimeError('node exited after camera initialization')
                        print('STARTUP_PASS '+name, flush=True)
                        break
                    time.sleep(.1)
                else: raise RuntimeError(name+' camera not initialized within 15 seconds')
            finally:
                if node is not None: terminate(node)
                terminate(camera)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    check(args.workspace.resolve(), args.out.resolve())
