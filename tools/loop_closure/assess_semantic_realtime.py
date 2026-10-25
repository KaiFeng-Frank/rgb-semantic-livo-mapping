"""Admission against budgets frozen before the live semantic replay."""
import argparse
import json
from pathlib import Path

from .assess_realtime import lines
from .realtime_replay import stats


def assess(control, online):
    cfg = json.loads(Path(__file__).with_name('semantic_realtime_protocol.json').read_text())
    b, o = [json.loads((d/'assessment.json').read_text()) for d in (control, online)]
    final = json.loads((online/'semantics/final.json').read_text())
    events = lines(online/'semantics/events.jsonl')
    maps, scans = lines(online/'semantic_maps.jsonl'), lines(online/'semantic_scans.jsonl')
    states = lines(online/'semantic_status.jsonl')
    geometry = lines(online/'status.jsonl')
    inputs = {r['stamp_ns']: r for r in lines(online/'semantic_inputs.jsonl')}
    sensor_inputs = {r['stamp_ns']: r['sent'] for r in lines(online/'inputs.jsonl')}
    frames = [e for e in events if e['event'] == 'keyframe']
    snapshots = [e for e in events if e['event'] == 'snapshot']
    revisions = [e for e in snapshots if e.get('revision') and e['revision'].get('changed_keyframes', 0)]
    map_delay = stats([(m['received']-m['graph_received_monotonic'])*1000 for m in maps])
    scan_delay = stats([(s['received']-inputs[s['stamp_ns']]['received'])*1000 for s in scans if s['stamp_ns'] in inputs])
    raw = [[p['stamp_ns'] for p in lines(d/'poses.jsonl') if p['kind'] == 'raw'] for d in (control, online)]
    graph = [r for r in lines(online/'worker.jsonl') if r.get('status') == 'updated']
    losses = ('dropped_full', 'dropped_stale', 'dropped_order', 'unanchored', 'invalid', 'graph_invalid', 'capacity_reached')
    parity = final.get('rebuild_parity', {})
    checks = dict(
        complete_real_sequence=min(b['duration_s'], o['duration_s']) >= cfg['minimum_sensor_duration_s'],
        paced_1x=all(a['replay_wall_s']/a['duration_s'] < 1.02 for a in (b, o)),
        same_source=b['source_payload_sha256'] == o['source_payload_sha256'],
        same_pose_timestamps=raw[0] == raw[1],
        all_pose_outputs=o['missing_corrected'] == o['missing_images_after_initialization'] == 0,
        initialization_only=o['warmup_prefix_images'] <= cfg['maximum_initialization_images'],
        pose_latency=o['corrected_end_to_end']['p95_ms'] <= cfg['pose_p95_ms'] and o['corrected_end_to_end']['max_ms'] <= cfg['pose_max_ms'],
        map_latency=bool(map_delay and map_delay['p95_ms'] <= cfg['semantic_graph_receipt_to_map_publication_p95_ms'] and
                         map_delay['max_ms'] <= cfg['semantic_graph_receipt_to_map_publication_max_ms']),
        scan_latency=bool(scan_delay and scan_delay['p95_ms'] <= cfg['semantic_cloud_receipt_to_scan_publication_p95_ms'] and
                          scan_delay['max_ms'] <= cfg['semantic_cloud_receipt_to_scan_publication_max_ms']),
        no_semantic_drops=all(final[k] == 0 for k in losses),
        every_keyframe_classified=final['processed'] == final['received'] == len(inputs) == len(graph),
        every_semantic_scan_delivered=len({s['stamp_ns'] for s in scans}) == len(inputs),
        nonempty_static_map=final['map_info']['summary']['map_voxels'] > 0 and final['map_info']['summary']['stored_points'] > 0,
        live_online_loops=bool(graph and graph[-1]['loop_count'] > 0),
        actual_semantic_revisions=bool(revisions),
        graph_and_outputs_healthy=all(not s.get('error') and not s.get('output_error') and not s.get('trajectory_error') for s in geometry),
        no_graph_overload=all(s['ipc']['queue_full'] == s['ipc']['stale'] == s['ipc']['late_results'] == 0 for s in geometry),
        semantic_healthy=not final['failed'] and all(not s['failed'] and not s['output_error'] for s in states),
        final_queue_empty=final['pending'] == 0,
        latest_revision_applied=final['map_info']['graph_version'] == final['latest_graph_version'] and
                               final['map_info']['graph_stamp_ns'] == final['latest_graph_stamp_ns'],
        # verify_rebuild has an explicit passed/admitted boolean, never inferred
        # from lack of an exception or from a map merely being nonempty.
        fresh_rebuild_parity=bool(parity.get('passed', parity.get('ok', False))),
    )
    return dict(admitted=all(checks.values()), checks=checks, protocol=cfg,
        control=b, online=o, final=final, graph_keyframes=len(graph), accepted_loops=graph[-1]['loop_count'],
        semantic_revisions=len(revisions), map_publication_ms=map_delay, semantic_scan_ms=scan_delay,
        map_source_to_received_ms=stats([(m['received']-sensor_inputs[m['graph_stamp_ns']])*1000
                                         for m in maps if m['graph_stamp_ns'] in sensor_inputs]),
        inference_ms=stats([f['infer_ms'] for f in frames]),
        revision_job_ms=stats([s['job_ms'] for s in revisions]),
        all_fusion_job_ms=stats([s['job_ms'] for s in snapshots]),
        accuracy_evaluated=False, semantic_inference_included=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'online', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    result = assess(args.control, args.online)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(admitted=result['admitted'], checks=result['checks']), indent=2))
    if not result['admitted']:
        raise SystemExit(1)
