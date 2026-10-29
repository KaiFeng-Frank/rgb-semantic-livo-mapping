"""Freeze full-rate CAM_FRONT timestamps for the eight fixed development scenes."""
import argparse
from pathlib import Path
import numpy as np
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import DATA,records,digest


def prepare(index_path,out):
    index=read(index_path);names=index['independent_dev_added'];out.mkdir(parents=True,exist_ok=False)
    sample_scene={r['sample_token']:name for name in names for r in index['sequences'][name]}
    keyframes={r['camera_token']:r['frame'] for name in names for r in index['sequences'][name]}
    scenes={n:[] for n in names}
    for r in records(DATA/'v1.0-trainval/sample_data.json'):
        name=sample_scene.get(r['sample_token'])
        if name is None or not (r['filename'].startswith('samples/CAM_FRONT/') or r['filename'].startswith('sweeps/CAM_FRONT/')):continue
        rows=index['sequences'][name]
        if not rows[0]['camera_timestamp_us']<=r['timestamp']<=rows[-1]['camera_timestamp_us']:continue
        scenes[name].append(dict(token=r['token'],timestamp_us=r['timestamp'],path=r['filename'],
            keyframe=keyframes.get(r['token']),width=r['width'],height=r['height']))
    counts={};requested=[]
    for name,frames in scenes.items():
        frames.sort(key=lambda r:r['timestamp_us']);times=np.array([r['timestamp_us'] for r in frames])
        if np.any(np.diff(times)<=0):raise ValueError('Nonmonotonic image stream')
        if {r['keyframe'] for r in frames if r['keyframe'] is not None}!={r['frame'] for r in index['sequences'][name]}:raise ValueError('Missing keyframes')
        counts[name]=dict(frames=len(frames),keyframes=sum(r['keyframe'] is not None for r in frames),
            median_dt_ms=float(np.median(np.diff(times))/1000),max_dt_ms=float(np.diff(times).max()/1000))
        requested.extend(r['path'] for r in frames if r['keyframe'] is None)
    save(out/'frames.json',dict(scenes=scenes,index_sha256=digest(index_path),counts=counts,
        selection='Exactly the eight independent development scenes already selected before raw evidence extraction; same first/last keyframe times, no GT/prediction selection'))
    save(out/'requested-paths.json',sorted(requested))
    save(out/'protocol.json',dict(source_index_sha256=digest(index_path),frames_sha256=digest(out/'frames.json'),
        data='Only past/intervening CAM_FRONT images. The original 2Hz LiDAR keyframes, point GT and scoring domains remain unchanged. New holdout stays closed.',
        hypothesis='Feeding real intervening images to the existing tracker improves identity continuity compared with subsampled tracking. This is extra temporal evidence, not a new network contribution.',
        controls=['Original keyframe-only BoT-SORT','Same BoT-SORT with all intervening causal camera frames'],
        detection='One frozen YOLO forward per image. Keyframe detections are shared exactly across arms. Neither arm uses ReID or a new detector.',
        tracking='Separate causal GMC histories; identical matching settings, nominal lost-frame buffers 15 keyframes/90 images with an additional strict 7.5s real-timestamp expiry in both arms. Refuse gaps over 750ms; each scene starts fresh.',
        scores='Full/camera point IDF1/AQ, coverage and switches at identical original keyframes; all logs, detector/GMC/tracker wall times and cold start. No semantic improvement claimed from identity alone.',
        gate='At least +1 pp camera point IDF1, no full-scan point IDF1 loss, positive majority of eight logs; otherwise preserve failure without threshold tuning.',
        limits=dict(max_new_image_bytes=700000000,max_network_bytes=1000000000,minimum_free_bytes=4000000000),
        source='https://huggingface.co/datasets/Xiaodong/Nuscenes-v1.0-trainval-CAM_FRONT_Sweeps/tree/11f15b33699f8a5fc8ad7ed042a715d17df146d6'))
    print(dict(counts=counts,missing_images=len(requested)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.index,a.out)
