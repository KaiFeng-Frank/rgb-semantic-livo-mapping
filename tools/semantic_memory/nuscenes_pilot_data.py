"""Small, reproducible nuScenes pilot index; no scan/label downloads or copies.

This adapter declares RAW keyframe geometry with rigid ego-pose alignment. It
does not label scans as deskewed and cannot make a deployable ROS checkpoint.
"""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/media/frank/Extreme SSD/rgb_lidar_shared_backbone/remote_snapshot/rgb_lidar_shared_backbone')
DATA = ROOT / 'datasets/nuscenes/s0_trainval'
SPLITS = Path('/home/frank/rgb_lidar_local_runtime/.venv/lib/python3.11/site-packages/nuscenes/utils/splits.py')
# Official coarse lidarseg IDs minus one, in the frozen PTv3 head order.
LABEL_MAP = np.array([-1,-1,6,6,6,-1,6,-1,-1,0,-1,-1,7,-1,1,2,2,3,4,-1,-1,5,8,9,10,11,12,13,14,-1,15,-1], dtype=np.int64)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def records(path):
    """Stream a JSON array with bounded buffers, including multiline records."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buf, pos, eof, started = '', 0, False, False
        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if not started and pos < len(buf):
                if buf[pos] != '[':
                    raise ValueError('Expected a JSON array')
                started, pos = True, pos + 1
                continue
            if pos < len(buf) and buf[pos] == ',':
                pos += 1
                continue
            if pos < len(buf) and buf[pos] == ']':
                return
            if pos < len(buf):
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                else:
                    pos = end
                    yield obj
                    continue
            if eof:
                raise ValueError('Truncated JSON array')
            chunk = stream.read(1024 * 1024)
            buf, pos, eof = buf[pos:] + chunk, 0, not bool(chunk)
            if len(buf) > 16 * 1024 * 1024:
                raise ValueError('Unexpectedly large metadata record')


def official_splits(path=SPLITS):
    # Read literal lists without importing the devkit's heavy dependencies.
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('train_detect', 'train_track', 'val'):
                result[name] = set(ast.literal_eval(node.value))
    return dict(train=result['train_detect'] | result['train_track'], test=result['val'])


def transform(row):
    from pyquaternion import Quaternion
    t = np.eye(4)
    t[:3, :3] = Quaternion(row['rotation']).rotation_matrix
    t[:3, 3] = row['translation']
    return t


def build_index(out, train_scenes=6, dev_scenes=2, test_scenes=6, *, selection_salt=None, excluded_scenes=()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = DATA/'v1.0-trainval'
    scenes = {r['name']: r for r in json.loads((metadata/'scene.json').read_text())}
    official = official_splits()
    manifests = {}
    by_scene = defaultdict(list)
    for part, name in (('train', 'train'), ('test', 'val')):
        path = ROOT/f'runs/s0_formal/{name}_manifest.json'
        manifest = json.loads(path.read_text())
        manifests[part] = digest(path)
        if set(manifest['official_split_scene_names']) != official[part]:
            raise ValueError('Source manifest differs from official split')
        for row in manifest['samples']:
            if row['scene_name'] not in official[part]:
                raise ValueError('Foreign sample in source split')
            by_scene[row['scene_name']].append(row)
    # Select using log tokens only, before reading labels or predictions. Never
    # select adjacent scenes from the same collection log into different parts.
    def choose(names, count, salt, excluded):
        groups = defaultdict(list)
        for name in names:
            log = scenes[name]['log_token']
            if log not in excluded:
                groups[log].append(name)
        ranked = sorted(groups, key=lambda x: hashlib.sha256((salt+x).encode()).hexdigest())
        if len(ranked) < count:
            raise ValueError('Insufficient independent logs')
        chosen = [sorted(groups[log])[0] for log in ranked[:count]]
        return chosen, set(ranked[:count])
    excluded_logs = {scenes[n]['log_token'] for n in excluded_scenes}
    salt = selection_salt or 'memory-pilot-20261025'
    test, test_logs = choose(official['test'], test_scenes, salt+'-test-', excluded_logs)
    train_dev, train_dev_logs = choose(official['train'], train_scenes+dev_scenes,
                                        salt+'-train-', test_logs | excluded_logs)
    parts = dict(train=train_dev[:train_scenes], dev=train_dev[train_scenes:], test=test)
    selected = {name: sorted(by_scene[name], key=lambda x:x['timestamp_us']) for names in parts.values() for name in names}
    tokens = {r['sample_data_token'] for rows in selected.values() for r in rows}
    sample_tokens = {r['sample_token'] for rows in selected.values() for r in rows}
    print(json.dumps(dict(stage='selected', parts=parts, frames={p:sum(len(selected[n]) for n in ns) for p,ns in parts.items()})), flush=True)
    data = {r['token']: r for r in records(metadata/'sample_data.json') if r['token'] in tokens}
    if set(data) != tokens:
        raise ValueError('Missing sensor metadata')
    pose_tokens = {r['ego_pose_token'] for r in data.values()}
    poses = {r['token']: r for r in records(metadata/'ego_pose.json') if r['token'] in pose_tokens}
    calibration = {r['token']:r for r in json.loads((metadata/'calibrated_sensor.json').read_text())}
    sensor = {r['token']:r for r in json.loads((metadata/'sensor.json').read_text())}
    categories = {r['token']:r for r in json.loads((metadata/'category.json').read_text())}
    instances = {r['token']:r for r in json.loads((metadata/'instance.json').read_text())}
    annotations = defaultdict(list)
    for a in records(metadata/'sample_annotation.json'):
        if a['sample_token'] not in sample_tokens:
            continue
        cat = categories[instances[a['instance_token']]['category_token']]
        label = int(LABEL_MAP[cat['index']])
        if label < 0:
            continue
        annotations[a['sample_token']].append(dict(instance=a['instance_token'], label=label,
            translation=a['translation'], rotation=a['rotation'], size=a['size'],
            annotated_lidar_points=a['num_lidar_pts']))
    for name, rows in selected.items():
        for i, r in enumerate(rows):
            sd = data[r['sample_data_token']]
            cal = calibration[sd['calibrated_sensor_token']]
            if sensor[cal['sensor_token']]['channel'] != 'LIDAR_TOP' or not sd['is_key_frame']:
                raise ValueError('Invalid sensor/keyframe metadata')
            if sd['filename'] != r['scan_path'] or sd['timestamp'] != r['timestamp_us']:
                raise ValueError('Source sensor metadata mismatch')
            for field, hashfield in (('scan_path','scan_sha256'),('label_path','label_sha256')):
                if digest(DATA/r[field]) != r[hashfield]:
                    raise ValueError('Dataset file hash mismatch')
            r['frame'] = i
            r['transform'] = (transform(poses[sd['ego_pose_token']]) @ transform(cal)).tolist()
            r['objects'] = annotations[r['sample_token']]
            for key in ('image_path','image_sha256','camera_token','camera_timestamp_us','body_forward_lidar'):
                r.pop(key, None)
    result = dict(schema=1, dataset='nuScenes-v1.0-trainval', data_root=str(DATA),
        geometry='raw_keyframe_sensor_with_rigid_ego_alignment_not_deskewed',
        intensity='raw remission divided by 255; extractor intensity_scale=1.0',
        supervision='official native lidarseg labels; both learned arms use identical training frames',
        split_source_sha256=digest(SPLITS), source_manifest_sha256=manifests,
        selection='one lexicographically first scene per hashed collection log; no prediction/label selection',
        parts=parts, scene_logs={n:scenes[n]['log_token'] for n in selected}, sequences=selected,
        label_map=LABEL_MAP.tolist(), selection_seed=selection_salt or 20261025,
        official_reference='https://www.nuscenes.org/lidar-segmentation')
    (out/'index.json').write_text(json.dumps(result, separators=(',',':'))+'\n')
    print(json.dumps(dict(stage='index_ready', path=str(out/'index.json'))), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args=p.parse_args()
    build_index(args.out)
