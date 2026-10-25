"""Check incremental optimization on archived seq07 edges without new data/GT.

This tests the optimizer transition only. The existing accepted loop is replayed
at its original current keyframe; it is not a new online retrieval evaluation.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .backend import optimize
from .common import REPO, S, digest, rigid_pose
from .online import configuration, IncrementalGraph


def run(times_file, out):
    source = REPO/'results/loop_closure_seq07_20261024'
    archive = json.loads((source/'loops.json').read_text())
    times = np.asarray(json.loads(times_file.read_text())['times_ns'], dtype=np.int64)*1e-9
    traj = S.TrajInterp(source/'baseline_imu_tum.txt')
    if not traj.valid(times).all():
        raise ValueError('timestamps outside archived odometry')
    # The archived ICP measurements are in LiDAR frames, so retain those frames
    # here. Live exporter/graph use IMU frames consistently on both ends instead.
    poses = np.array([rigid_pose(traj.query_one(t)[0] @ S.T_I_L) for t in times])
    edges = [(r['i'], r['j'], np.array(r['T_historical_current']))
             for r in archive['attempts'] if r['accepted']]
    cfg = configuration()
    graph, costs = IncrementalGraph(cfg), []
    for j, pose in enumerate(poses):
        start = time.monotonic()
        graph.append(pose)
        for a, b, measurement in edges:
            if b == j:
                graph.add_loop(a, b, measurement)
        costs.append((time.monotonic()-start)*1000)
    incremental = graph.poses()
    batch, batch_stats = optimize(poses, edges, cfg)
    error = float(graph.isam.getFactorsUnsafe().error(graph.isam.calculateEstimate()))
    difference = float(np.linalg.norm(incremental[:, :3, 3]-batch[:, :3, 3], axis=1).max())
    result = dict(scope='archived odometry and accepted edges; no new retrieval, raw sensors or GT',
                  nodes=len(poses), loops=len(edges), graph_error=error, batch=batch_stats,
                  position_difference_to_batch_max_m=difference,
                  update_median_ms=float(np.median(costs)), update_max_ms=float(max(costs)),
                  inputs_sha256={p.name: digest(p) for p in
                                 (source/'baseline_imu_tum.txt', source/'loops.json', times_file)},
                  passed=bool(difference < .001 and error-batch_stats['error_after'] < 1e-5))
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('incremental graph regression')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--times', type=Path, default=REPO/'results/online_backend_20261025/keyframe_times_ns.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.times, args.out)
