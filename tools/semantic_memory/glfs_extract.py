"""One shared frozen-encoder extraction, retaining actual individual history observations."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from tools.instance_memory.data import read, save
from tools.instance_memory.metrics import identity_truth
from .nuscenes_pilot_data import digest, LABEL_MAP
from .feature_supervision_audit import voxel_truth
from .glfs_baseline import observation_geometry, verify_sources


def extract(root, data, limit=None):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
    from semantic_repair import CurrentAnchoredRepair, RepairConfig, JointRepair
    from semantic_memory import MemoryConfig
    from ptv3_worker import Segmenter
    from ptv3_loader_verified import serialization_policy
    from .repair_pilot import anchor_model, call
    from .nuscenes_pilot import current_output
    from .repair_supervision import current_token_input
    from .chain_trace import TracedEvidence, FusionCapture
    torch.set_num_threads(2)
    torch.manual_seed(20261031)
    torch.cuda.set_per_process_memory_fraction(.18)
    protocol = read(root/'protocol.json')
    verify_sources(root, protocol)
    for name, expected in protocol['input_hashes'].items():
        if digest(root/'inputs'/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    if digest(Path(os.environ['PTV3_CHECKPOINT'])) != protocol['backbone_sha256']:
        raise ValueError('Wrong frozen backbone')
    cache = root/('smoke-cache' if limit else 'cache')
    cache.mkdir(exist_ok=False)
    ck = torch.load(root/'inputs/repair-reference.pt', map_location='cpu', weights_only=True)
    anchor = anchor_model(root/'inputs/anchor.pt')
    model = CurrentAnchoredRepair(MemoryConfig(**ck['config']), RepairConfig(**ck['repair']), device='cuda').eval()
    model.fusion.load_state_dict(ck['fusion'], strict=True)
    model.memory = TracedEvidence(model.cfg, 'cuda', model.repair.moving_classes)
    capture = FusionCapture(model.fusion)
    inp = {}
    handle = model.fusion.net.register_forward_pre_hook(lambda _, a: inp.update(z=a[0].detach()))
    heads = []
    for seed in protocol['seeds']:
        path = root/f'inputs/current-{seed}.pt'
        if digest(path) != protocol['current_checkpoints'][str(seed)]:
            raise ValueError('Current classifier changed')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        head = JointRepair(model.cfg, model.repair).cuda().eval()
        head.load_state_dict(saved['fusion'], strict=True)
        head.requires_grad_(False)
        heads.append(head)
    backbone = Segmenter(half=True, shuffle=False, intensity_scale=1.)
    index = read(root/'inputs/index.json')
    manifest = dict(protocol_sha256=digest(root/'protocol.json'), files=[], geometry=[],
                    serialization=serialization_policy(backbone.model), runtime=dict(torch=torch.__version__, numpy=np.__version__),
                    total_bytes=0, elapsed_s=0, normal_geometry_s=0, current_path_parity_checks=0, deployable=False,
                    latency_scope='Extraction includes trace instrumentation and concurrent TASeg; not an online latency measurement')
    start = time.perf_counter()
    with torch.no_grad():
        for part, scenes in protocol['parts'].items():
            for scene in scenes:
                model.reset()
                previous = None
                rows = index['sequences'][scene]
                targets = read(root/f'inputs/targets/{scene}.targets.json.gz')['frames']
                if len(rows) != len(targets):
                    raise ValueError('Incomplete fixed evaluation annotations')
                origin = np.linalg.inv(np.asarray(rows[0]['transform']))
                (cache/scene).mkdir()
                for k, (row, target) in enumerate(zip(rows, targets)):
                    path, label_path = data/row['scan_path'], data/row['label_path']
                    if digest(path) != row['scan_sha256'] or digest(label_path) != row['label_sha256']:
                        raise ValueError('Source differs from fixed original index')
                    scan = np.fromfile(path, dtype='<f4').reshape(-1, 5)[:, :4].copy()
                    scan[:, 3] /= 255.
                    r = backbone.segment(scan, return_features=True)
                    raw = {key:r[key] for key in ('xyz', 'features', 'logits')}
                    raw.update(transform=torch.tensor(origin@np.asarray(row['transform']), device='cuda', dtype=torch.float32),
                               stamp=row['timestamp_us']/1e6, frame=k)
                    a = current_output(anchor, raw)['logits']
                    out = call(model, raw, a)
                    query = out['query_indices']
                    zc = current_token_input(inp['z'])
                    current = torch.stack([a[query]+model.repair.residual_limit*torch.tanh(h.net(zc)) for h in heads])
                    if k in (0, len(rows)//2, len(rows)-1):
                        f = r['features'][query]
                        empty = dict(features=f[:, None], probabilities=a[query].float().softmax(-1)[:, None],
                            delta=f.new_zeros((len(query), 1, 3)), age=f.new_zeros((len(query), 1)),
                            valid=torch.zeros((len(query), 1), device='cuda', dtype=torch.bool))
                        for seed, head in enumerate(heads):
                            if not torch.equal(head(f, a[query], empty), current[seed]):
                                raise AssertionError('Cached current path differs from independent native forward')
                            manifest['current_path_parity_checks'] += 1
                    candidate = capture.value[2]
                    n = len(query)
                    historical = np.zeros((n, 4, 16), np.float32)
                    depth, angle, valid = np.zeros((n, 5), np.float32), np.zeros((n, 5), np.float32), np.zeros((n, 5), bool)
                    source_index = np.full((n, 4), -1, np.int32)
                    tq = time.perf_counter()
                    dep, ang, stats = observation_geometry(r['xyz'].cpu().numpy())
                    geometry_s = time.perf_counter()-tq
                    manifest['normal_geometry_s'] += geometry_s
                    qi = query.cpu().numpy()
                    depth[:, 0], angle[:, 0], valid[:, 0] = dep[qi], ang[qi], True
                    if previous is not None and len(model.memory.trace['record_source']):
                        trace = model.memory.trace
                        indices = trace['record_source'][trace['shortlist_index']].cpu().numpy()
                        mask = candidate['valid'].cpu().numpy()
                        width = mask.shape[1]
                        if not torch.equal(torch.from_numpy(previous['prob_half'][indices]).cuda(), candidate['probabilities']):
                            raise AssertionError('Saved source does not match native retrieval')
                        historical[:, :width] = previous['logits'][indices]
                        depth[:, 1:width+1] = previous['depth'][indices]
                        angle[:, 1:width+1] = previous['angle'][indices]
                        valid[:, 1:width+1] = mask
                        source_index[:, :width] = np.where(mask, indices, -1)
                        if not previous['stamp'] < raw['stamp']:
                            raise ValueError('Future history')
                    base = r['logits'].float().cpu().numpy()
                    previous = dict(logits=base, prob_half=r['logits'].float().softmax(-1).half().cpu().numpy(),
                                    depth=dep, angle=ang, stamp=raw['stamp'])
                    # Evaluation labels are joined only after all inputs and predictions exist.
                    inverse = r['inverse'].cpu().numpy().astype(np.int32)
                    gt = np.asarray(target['gt'], np.int8)
                    np.testing.assert_array_equal(gt, LABEL_MAP[np.fromfile(label_path, dtype=np.uint8)])
                    identity = identity_truth(target).astype(np.int32)
                    y, _ = voxel_truth(inverse, gt, identity, len(base))
                    camera = np.zeros(len(gt), bool)
                    camera[target.get('camera_in_view_indices', [])] = True
                    arrays = dict(current=current.cpu().numpy(), history=historical, depth=depth, angle=angle, valid=valid,
                                  query=qi.astype(np.int32), anchor=a.float().cpu().numpy(), baseline=base.argmax(1).astype(np.int8),
                                  inverse=inverse, point_label=gt, identity=identity, camera=camera,
                                  label=y[qi].astype(np.int8), source_index=source_index)
                    path = cache/scene/f'{k:03d}.npz'
                    np.savez(path, **arrays)
                    manifest['files'].append(dict(part=part, scene=scene, log=index['scene_logs'][scene], frame=k,
                        stamp=raw['stamp'], path=str(path.relative_to(cache)), sha256=digest(path), queries=n,
                        with_history=int(valid[:, 1:].any(1).sum())))
                    manifest['geometry'].append(dict(scene=scene, frame=k, seconds=geometry_s, **stats))
                    manifest['total_bytes'] += path.stat().st_size
                    manifest['elapsed_s'] = time.perf_counter()-start
                    if limit and len(manifest['files']) >= limit:
                        save(cache/'manifest.json', manifest)
                        print('smoke complete', len(manifest['files']), flush=True)
                        return
                save(cache/'progress.json', dict(completed_frames=len(manifest['files']), scene=scene, elapsed_s=manifest['elapsed_s']))
                print('extracted', part, scene, len(manifest['files']), 'frames', flush=True)
    handle.remove()
    capture.close()
    if len(manifest['files']) != 963:
        raise AssertionError('Incomplete fixed split')
    save(cache/'manifest.json', manifest)
    print('extraction complete', manifest['elapsed_s'], 'seconds', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    extract(a.root, a.data, a.limit)
