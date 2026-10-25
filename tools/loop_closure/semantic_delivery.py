"""Match each semantic publication to its own actual cloud reception, once."""
from collections import defaultdict


def match_maps(infos, clouds):
    available = defaultdict(list)
    for cloud in sorted(clouds, key=lambda c: c['received']):
        key = (cloud['stamp_ns'], cloud['sha256'], cloud['points'], cloud['frame_id'])
        available[key].append(cloud)
    matched = []
    for info in sorted(infos, key=lambda m: m['publication_id']):
        key = (info['graph_stamp_ns'], info['map_sha256'], info['map_points'], 'map')
        candidates = available[key]
        # A prior snapshot can have the same displayed bytes. Its reception
        # must not be reused, or predate the current snapshot's completion.
        while candidates and candidates[0]['received'] < info['ready_monotonic']:
            candidates.pop(0)
        matched.append((info, candidates.pop(0) if candidates else None))
    return matched
