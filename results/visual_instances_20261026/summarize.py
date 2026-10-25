"""Recompute runtime statistics from the archived subscriber observations."""
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def stats(values):
    values = sorted(values)
    if not values:
        return None
    def percentile(q):
        at = (len(values)-1)*q
        lo = int(at)
        hi = min(lo+1, len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(at-lo)
    return dict(n=len(values), median_ms=percentile(.5), p95_ms=percentile(.95),
                p99_ms=percentile(.99), max_ms=values[-1])


def main():
    results = {}
    for directory in sorted((ROOT/'hku').iterdir()):
        if not directory.is_dir():
            continue
        assessment = json.loads((directory/'assessment.json').read_text())
        sent = {r['stamp_ns']: r['sent'] for r in rows(directory/'inputs.jsonl.gz')}
        corrected = {r['stamp_ns']: r['received'] for r in rows(directory/'poses.jsonl.gz')
                     if r['kind'] == 'corrected'}
        latency = stats([(received-sent[stamp])*1000 for stamp, received in corrected.items()])
        for key, value in latency.items():
            assert abs(value-assessment['corrected_end_to_end'][key]) < 1e-7, (directory, key)
        vision = directory/'vision/events.jsonl.gz'
        events = rows(vision) if vision.exists() else []
        missing = sorted(set(sent)-set(corrected))
        results[directory.name] = dict(
            input_images=len(sent), initialized_pose_outputs=len(corrected),
            missing_images_after_initialization=assessment['missing_images_after_initialization'],
            missing_image_offsets_s=[(stamp-min(sent))/1e9 for stamp in missing],
            corrected_pose_latency=latency, semantic_scan_outputs=assessment['semantic_scan_outputs'],
            visual_processed=len(events), visual_frames_with_instances=sum(r['instances'] > 0 for r in events),
            visual_receipt_to_publish=stats([r['receipt_to_publish_ms'] for r in events]),
            instance_cloud_outputs=assessment['instance_cloud_outputs'],
            associated_points=assessment['associated_points'],
            source_payload_sha256=assessment['source_payload_sha256'])
    (ROOT/'runtime-summary.json').write_text(json.dumps(results, indent=2)+'\n')
    for name, row in results.items():
        print(name, json.dumps(row['corrected_pose_latency']),
              'missing', row['missing_images_after_initialization'])


if __name__ == '__main__':
    main()
