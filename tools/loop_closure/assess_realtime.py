"""Evaluate paired recorded 1x replay against the frozen realtime budgets."""
import argparse
import json
from pathlib import Path

import numpy as np

from .realtime_replay import stats


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assess(baseline, online):
    cfg = json.loads(Path(__file__).with_name('realtime_protocol.json').read_text())
    arms = [json.loads((path/'assessment.json').read_text()) for path in (baseline, online)]
    b, o = arms
    raw_times = [[r['stamp_ns'] for r in lines(path/'poses.jsonl') if r['kind'] == 'raw']
                 for path in (baseline, online)]
    inputs = lines(online/'inputs.jsonl')
    stamps = [r['stamp_ns'] for r in inputs]
    missing = sorted(s for s in stamps if s >= min(raw_times[1]) and s not in set(raw_times[1]))
    states = lines(online/'status.jsonl')
    worker = [r for r in lines(online/'worker.jsonl') if r.get('status') == 'updated']
    sent = {r['stamp_ns']: r['sent'] for r in inputs}
    maps = lines(online/'maps.jsonl')
    checks = dict(
        input_counts=all(a['input_counts'] == dict(image=cfg['expected_images'], lidar=cfg['expected_lidar_scans'],
                                                 imu=cfg['expected_imu']) for a in arms),
        same_source=b['source_payload_sha256'] == o['source_payload_sha256'],
        same_frontend_timestamps=raw_times[0] == raw_times[1],
        integer_sensor_time_pairing=all(a['matched_sensor_stamps'] == a['frontend_poses'] for a in arms),
        no_backend_output_loss=o['missing_corrected'] == cfg['missing_backend_outputs'],
        only_terminal_frame_waiting=missing in ([], [stamps[-1]]),
        paced_1x=all(a['replay_wall_s']/a['duration_s'] <= cfg['replay_wall_ratio_max'] for a in arms),
        frontend_p95=all(a['frontend_end_to_end']['p95_ms'] <= cfg['frontend_end_to_end_p95_budget_ms'] for a in arms),
        frontend_max=all(a['frontend_end_to_end']['max_ms'] <= cfg['frontend_end_to_end_max_budget_ms'] for a in arms),
        corrected_p95=o['corrected_end_to_end']['p95_ms'] <= cfg['corrected_end_to_end_p95_budget_ms'],
        corrected_max=o['corrected_end_to_end']['max_ms'] <= cfg['corrected_end_to_end_max_budget_ms'],
        backend_added_p99=o['backend_added']['p99_ms'] <= cfg['backend_added_p99_budget_ms'],
        worker_and_output_healthy=all(not s['error'] and not s.get('output_error') for s in states),
        no_overload=all(s['ipc']['queue_full'] == s['ipc']['stale'] == s['ipc']['late_results'] == 0 for s in states),
        final_queue_drained=states[-1]['pending'] == 0 and states[-1]['ipc']['completed'] == states[-1]['ipc']['submitted'],
        capacity_not_reached=all(s.get('status') != 'capacity_reached' for s in states),
        live_loops=bool(worker and worker[-1]['loop_count'] > 0),
        live_map_and_path=bool(maps and o['paths']),
    )
    return dict(admitted=all(checks.values()), scope=cfg['scope'], protocol=cfg, checks=checks,
                baseline=b, online=o, missing_image_stamps=missing,
                graph_keyframes=len(worker), accepted_loops=worker[-1]['loop_count'],
                worker_complete_keyframe_ms=stats([r['total_ms'] for r in worker]),
                map_source_to_received_ms=stats([(m['received']-sent[m['stamp_ns']])*1000 for m in maps
                                                if m['stamp_ns'] in sent]),
                maximum_display_points=max(m['points'] for m in maps),
                accuracy_evaluated=False, semantic_inference_included=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'online', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.baseline, args.online)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(admitted=result['admitted'], checks=result['checks']), indent=2))
    if not result['admitted']:
        raise SystemExit(1)
