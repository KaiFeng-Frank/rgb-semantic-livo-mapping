"""Freeze an outcome-independent, complete-scene paired diagnostic subset."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import time

ROOT = Path('/home/wuyou/stocc_reference')
SALT = 'stocc-memory-diagnostic-20261031-v1'


def records(path):
    decoder = json.JSONDecoder()
    with path.open() as stream:
        buffer, position, eof, started = '', 0, False, False
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started and position < len(buffer):
                assert buffer[position] == '['
                started, position = True, position + 1
                continue
            if position < len(buffer) and buffer[position] == ',':
                position += 1
                continue
            if position < len(buffer) and buffer[position] == ']':
                return
            if position < len(buffer):
                try:
                    row, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                else:
                    position = end
                    yield row
                    continue
            if eof:
                raise ValueError('Truncated JSON array')
            chunk = stream.read(1024 * 1024)
            buffer, position, eof = buffer[position:] + chunk, 0, not bool(chunk)


def main():
    out = ROOT/'history-diagnostic'
    out.mkdir(exist_ok=True)
    assert not (out/'protocol.json').exists(), 'Preserve the frozen protocol'
    data = ROOT/'data/nuscenes'
    valid = set(json.loads((ROOT/'camera-plan.json').read_text())['scenes'])
    groups = defaultdict(list)
    for row in json.loads((data/'v1.0-trainval/scene.json').read_text()):
        if row['name'] in valid:
            groups[row['log_token']].append(row['name'])
    logs = sorted(groups, key=lambda token: hashlib.sha256((SALT+token).encode()).hexdigest())[:16]
    chosen = [sorted(groups[log])[0] for log in logs]
    source = data/'bevdetv4-nuscenes_infos_global_val.pkl'
    with source.open('rb') as stream:
        infos = pickle.load(stream)
    infos['infos'] = [row for row in infos['infos'] if row['scene_name'] in chosen]
    subset = out/'subset.pkl'
    with subset.open('wb') as stream:
        pickle.dump(infos, stream)
    tokens = {row['token'] for row in infos['infos']}
    lidar = {row['sample_token']: row for row in records(data/'v1.0-trainval/sample_data.json')
             if row['sample_token'] in tokens and row['is_key_frame']
             and row['filename'].startswith('samples/LIDAR_TOP/')}
    assert set(lidar) == tokens and len(chosen) == len(logs) == 16
    frames = [dict(token=row['token'], scene=row['scene_name'], frame=int(row['global_idx']),
                   timestamp_us=row['timestamp'], lidar_token=lidar[row['token']]['token'],
                   lidar_path=lidar[row['token']]['filename'])
              for row in sorted(infos['infos'], key=lambda row: (row['scene_name'], row['global_idx']))]
    protocol = dict(
        schema=1, frozen=time.time(), selection_salt=SALT, scenes=chosen,
        scene_logs=dict(zip(chosen, logs)), available_validation_logs=len(groups),
        frames=frames, frame_count=len(frames),
        selection='Hash-ranked validation collection logs; lexicographically first complete scene per selected log. No labels or prediction scores used.',
        scope='Previously evaluated official validation pool; descriptive diagnostic, not a new blind test.',
        arms=['native_history', 'cold_history'],
        intervention='Reset inference feature/probability/uncertainty/visit/flow history to native initialization before each frame; preserve metric memories and native architecture.',
        evaluation=dict(
            point_population='Current LiDAR points within the official ego grid and camera-visible occupancy mask; same points for both arms. Use lidarseg semantic targets.',
            object_population='Official panoptic thing instances with at least 5 eligible points; GT identity used only offline.',
            success_correct_fraction=.5, event_horizon_seconds=2.,
            event_types=['first_observed', 'reappearance_after_observation_gap', 'cold_history_error_onset'],
            event_notes='Observation gap is not proof of occlusion; ~0.5s keyframe recovery delay is distinct from inference latency. Keep censored episodes in denominators.',
            primary=['point_mIoU16', 'corrected_points', 'harmed_points', 'object_reappearance_success', 'object_error_recovery_success', 'recovery_delay', 'model_call_p50_p95'],
            subgroups=['car','pedestrian'], confidence='2000 paired bootstrap draws over the 16 collection logs, seed 20261031.'),
        checks=['Strict identical weight loading', 'Native rerun prediction hashes match previous full validation',
                'Every cold-history attention receives current feature keys/values and no past flow',
                'All 16 scene-first predictions match between arms', 'Temporal metric history untouched by reset',
                'Panoptic class/point-count agreement with independent lidarseg labels'],
        decision='Positive mechanism evidence requires positive net point corrections and positive net recovery successes with no reappearance loss; inspect car/pedestrian subgroups and collection-log uncertainty. Mixed or sparse evidence does not authorize automatic PTv3 training.',
        minimum_recovery_evidence=dict(episodes=30, collection_logs=8),
        source_metadata_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        subset_metadata_sha256=hashlib.sha256(subset.read_bytes()).hexdigest(),
        checkpoint_sha256='68491f55a2208fad884e3b664d0e34b8311688376380f7722e30cac6bde5ffa0',
        no_training=True)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps(dict(scenes=chosen, frames=len(frames), logs=len(logs))), flush=True)


if __name__ == '__main__':
    main()
