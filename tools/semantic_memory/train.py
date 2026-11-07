#!/usr/bin/env python3
"""Train/evaluate causal fusion from frozen, raw NPZ observations only.

No dataset loader or backbone is imported. Example (also works with ``-m``)::

    python tools/semantic_memory/train.py train --train train.json --val val.json --out run
    python tools/semantic_memory/train.py evaluate --manifest test.json --checkpoint run/fusion.pt --out scores
    python tools/semantic_memory/train.py smoke --out synthetic-smoke

A manifest has schema=1, split=train|val|test, label_space=native16|semantic-kitti19,
feature_dim=64, backbone_sha256, model_config_sha256, source_fingerprint (SHA-256
of the frozen cache extractor/configuration), feature_contract, and sequences=[{id, frames:[{frame,
stamp, path, sha256?}]}]. Optional frame sha256 is verified against the NPZ bytes.
IDs are canonical two-digit KITTI sequence IDs. Paths are relative
to, and contained beneath, the manifest directory. Optional synthetic=true marks
test fixtures and propagates to a checkpoint that CANNOT be deployed.

feature_contract must be exactly {grid_size:0.05, intensity_scale:0.2,
feature_source:"unfused_decoder", geometry:"deskewed_sensor", label_space:...,
half:true, shuffle:false}.
Each NPZ contains xyz (N,3), features (N,64; float16/float32), logits (N,C),
transform (4,4; sensor-to-memory SE(3)), optional label (N; integers -1 or
0..C-1), and metadata: a scalar Unicode JSON string repeating schema,
feature_dim, label_space, backbone_sha256, model_config_sha256, source_fingerprint,
feature_contract, and synthetic (explicit boolean). This contract declares unfused decoder
features and raw head logits; extra arrays such as fused_features are rejected.
There is no implicit ontology conversion: native16 is NOT SemanticKITTI-19.

Train IDs: 00,01,02,04,05,06,10. Validation: 08. Test: legacy heldouts 07/09
or official test IDs 11..21. Test manifests are never accepted for training or
model selection. All frames are preflighted before the first optimizer update;
content hashes are checked again on every read. Clips are consecutive optimizer
batches; memory persists across clip boundaries and resets per sequence/epoch.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))
from semantic_memory import CausalSemanticMemory, MemoryConfig, checkpoint, training_loss


LABEL_SPACES = {'native16': 16, 'semantic-kitti19': 19}
TRAIN_IDS = frozenset(('00', '01', '02', '04', '05', '06', '10'))
VAL_IDS = frozenset(('08',))
TEST_IDS = frozenset(('07', '09', *(f'{i:02d}' for i in range(11, 22))))
META_KEYS = ('schema', 'feature_dim', 'label_space', 'backbone_sha256', 'model_config_sha256',
             'source_fingerprint', 'feature_contract', 'synthetic')


def feature_contract(label_space):
    return dict(grid_size=0.05, intensity_scale=0.2, feature_source='unfused_decoder',
                geometry='deskewed_sensor', label_space=label_space, half=True, shuffle=False)


def _valid_feature_contract(value, label_space):
    return (isinstance(value, dict) and value == feature_contract(label_space)
            and value.get('half') is True and value.get('shuffle') is False)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(text):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(text, object_pairs_hook=_object, parse_constant=invalid)


def _keys(value, required, optional=(), context='object'):
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be an object')
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f'{context} keys: missing={sorted(missing)}, unexpected={sorted(extra)}')


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a lowercase SHA-256 hex digest')
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _stamp(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('stamp must be a finite sensor timestamp')
    return float(value)


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass(frozen=True)
class Frame:
    frame: int
    stamp: float
    path: Path
    sha256: str


@dataclass(frozen=True)
class Sequence:
    id: str
    frames: tuple[Frame, ...]


@dataclass(frozen=True)
class Manifest:
    path: Path
    sha256: str
    split: str
    label_space: str
    feature_dim: int
    backbone_sha256: str
    model_config_sha256: str
    source_fingerprint: str
    feature_contract: dict
    synthetic: bool
    sequences: tuple[Sequence, ...]

    @property
    def metadata(self):
        return dict(schema=1, **{key: getattr(self, key) for key in META_KEYS if key != 'schema'})

    @property
    def ids(self):
        return [sequence.id for sequence in self.sequences]

    @property
    def frames(self):
        return [frame for sequence in self.sequences for frame in sequence.frames]

    @property
    def cache_fingerprint(self):
        rows = [(s.id, f.frame, f.stamp, f.path.relative_to(self.path.parent).as_posix(), f.sha256)
                for s in self.sequences for f in s.frames]
        return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def load_manifest(path, expected_split=None):
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()
    data = _json(content)
    required = ('schema', 'split', 'label_space', 'feature_dim', 'backbone_sha256',
                'model_config_sha256', 'source_fingerprint', 'feature_contract', 'sequences')
    _keys(data, required, ('synthetic',), 'manifest')
    if type(data['schema']) is not int or data['schema'] != 1:
        raise ValueError('Manifest schema must be 1')
    split = data['split']
    if split not in ('train', 'val', 'test') or (expected_split and split != expected_split):
        raise ValueError(f'Expected split {expected_split or "train/val/test"}, got {split!r}')
    space = data['label_space']
    if not isinstance(space, str) or space not in LABEL_SPACES:
        raise ValueError('Explicit label_space must be native16 or semantic-kitti19')
    if type(data['feature_dim']) is not int or data['feature_dim'] != 64:
        raise ValueError('Raw decoder feature_dim must be 64')
    for key in ('backbone_sha256', 'model_config_sha256', 'source_fingerprint'):
        _hash(data[key], key)
    if not _valid_feature_contract(data['feature_contract'], space):
        raise ValueError('feature_contract differs from the frozen unfused decoder contract')
    synthetic = data.get('synthetic', False)
    if type(synthetic) is not bool:
        raise ValueError('synthetic must be a boolean')
    if not isinstance(data['sequences'], list) or not data['sequences']:
        raise ValueError('Manifest needs nonempty sequences')
    sequences, ids, paths = [], set(), set()
    allowed = {'train': TRAIN_IDS, 'val': VAL_IDS, 'test': TEST_IDS}[split]
    for seq in data['sequences']:
        _keys(seq, ('id', 'frames'), context='sequence')
        sid = seq['id']
        if not isinstance(sid, str) or sid not in allowed:
            raise ValueError(f'Sequence {sid!r} is forbidden in split {split}; allowed={sorted(allowed)}')
        if sid in ids:
            raise ValueError(f'Duplicate sequence id: {sid}')
        ids.add(sid)
        if not isinstance(seq['frames'], list) or not seq['frames']:
            raise ValueError(f'Sequence {sid} needs nonempty frames')
        frames, last_frame, last_stamp = [], -1, -math.inf
        for item in seq['frames']:
            _keys(item, ('frame', 'stamp', 'path'), ('sha256',), context='frame')
            number, stamp = _integer(item['frame'], 'frame'), _stamp(item['stamp'])
            if number <= last_frame or stamp <= last_stamp:
                raise ValueError(f'Sequence {sid}: non-monotonic frame or sensor stamp')
            relative = item['path']
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise ValueError('Cache paths must be relative to the manifest')
            cache = (path.parent / relative).resolve(strict=True)
            if not cache.is_relative_to(path.parent) or cache.suffix != '.npz' or not cache.is_file():
                raise ValueError('Cache path must be an NPZ contained beneath the manifest directory')
            if cache in paths:
                raise ValueError(f'Duplicate cache path: {relative}')
            paths.add(cache)
            actual_hash = _digest(cache)
            if 'sha256' in item and _hash(item['sha256'], 'frame sha256') != actual_hash:
                raise ValueError(f'{cache}: published frame sha256 mismatch')
            frames.append(Frame(number, stamp, cache, actual_hash))
            last_frame, last_stamp = number, stamp
        sequences.append(Sequence(sid, tuple(frames)))
    return Manifest(path, hashlib.sha256(content).hexdigest(), split, space, 64,
                    data['backbone_sha256'], data['model_config_sha256'], data['source_fingerprint'],
                    data['feature_contract'], synthetic, tuple(sequences))


def load_frame(manifest, frame, max_points=200000):
    """Validate and detach one cache. Never recompute features or load pickles."""
    try:
        with frame.path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != frame.sha256:
                raise ValueError('Cache content changed after manifest preflight')
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as cache:
                _keys(dict.fromkeys(cache.files), ('xyz', 'features', 'logits', 'transform', 'metadata'),
                      ('label',), 'NPZ')
                if len(cache.files) != len(set(cache.files)):
                    raise ValueError('Duplicate NPZ arrays')
                meta = cache['metadata']
                if meta.shape != () or meta.dtype.kind != 'U':
                    raise ValueError('NPZ metadata must be a scalar Unicode JSON string')
                metadata = _json(meta.item())
                _keys(metadata, META_KEYS, context='NPZ metadata')
                # JSON equality alone would accept schema=true or synthetic=0.
                if (type(metadata['schema']) is not int or type(metadata['feature_dim']) is not int
                        or type(metadata['synthetic']) is not bool or metadata != manifest.metadata
                        or not _valid_feature_contract(metadata['feature_contract'], manifest.label_space)):
                    raise ValueError('NPZ metadata/schema/backbone/model_config_sha256/source/feature_contract mismatch')
                arrays = {key: cache[key] for key in ('xyz', 'features', 'logits', 'transform')}
                xyz = arrays['xyz']
                if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) > max_points:
                    raise ValueError('Invalid xyz shape or max_points exceeded')
                n, classes = len(xyz), LABEL_SPACES[manifest.label_space]
                shapes = dict(xyz=(n, 3), features=(n, 64), logits=(n, classes), transform=(4, 4))
                for name, array in arrays.items():
                    if array.shape != shapes[name] or array.dtype.kind != 'f' or not np.isfinite(array).all():
                        raise ValueError(f'Invalid/nonfinite {name} shape or dtype')
                    if name in ('features', 'logits') and array.dtype not in (np.dtype('float16'), np.dtype('float32')):
                        raise ValueError(f'{name} must be raw float16 or float32 arrays')
                    if name == 'features' and np.any(np.abs(array) > np.finfo(np.float16).max):
                        raise ValueError('features exceed finite float16 memory storage range')
                    # Core operates in float32 for geometry and logits.
                    if np.any(np.abs(array) > np.finfo(np.float32).max):
                        raise ValueError(f'{name} exceeds finite float32 range')
                transform = arrays['transform'].astype(np.float32)
                rot = transform[:3, :3]
                if (not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4, rtol=0)
                        or not np.isclose(np.linalg.det(rot), 1, atol=1e-4, rtol=0)
                        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6, rtol=0)):
                    raise ValueError('transform must be sensor-to-memory SE(3)')
                label = cache['label'] if 'label' in cache else np.full(n, -1, np.int64)
                if label.shape != (n,) or label.dtype.kind not in 'iu':
                    raise ValueError('label must be an integer array of shape (N,)')
                if np.any(label < -1) or np.any(label >= classes):
                    raise ValueError('label outside explicit label_space; use -1 for ignored points')
                arrays['label'] = label.astype(np.int64)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f'{frame.path}: {error}') from error
    return {name: torch.from_numpy(np.ascontiguousarray(array)).detach()
            for name, array in arrays.items()}


def training_pair(train, val):
    if train.split != 'train' or val.split != 'val':
        raise ValueError('Training requires train and val manifests; test training is forbidden')
    for key in ('feature_dim', 'label_space', 'backbone_sha256', 'model_config_sha256', 'source_fingerprint',
                'feature_contract', 'synthetic'):
        if getattr(train, key) != getattr(val, key):
            raise ValueError(f'Train/val {key} mismatch')
    if set(train.ids) & set(val.ids):
        raise ValueError('Train/val sequences must be disjoint')
    if ({f.path for f in train.frames} & {f.path for f in val.frames}
            or {f.sha256 for f in train.frames} & {f.sha256 for f in val.frames}):
        raise ValueError('Train/val cache paths and content must be disjoint')
    train_files = {(f.path.stat().st_dev, f.path.stat().st_ino) for f in train.frames}
    if train_files & {(f.path.stat().st_dev, f.path.stat().st_ino) for f in val.frames}:
        raise ValueError('Train/val cache hard links must be disjoint')


def preflight(manifest, cfg, require_labels=False):
    if cfg.feature_dim != 64 or cfg.classes != LABEL_SPACES[manifest.label_space]:
        raise ValueError('MemoryConfig differs from cache feature/class dimensions')
    supervised = 0
    for frame in manifest.frames:
        raw = load_frame(manifest, frame, cfg.max_points)
        world = raw['xyz'].float() @ raw['transform'][:3, :3].float().T + raw['transform'][:3, 3].float()
        grid = torch.floor(world / cfg.cell_size)
        if not bool(torch.isfinite(world).all()) or bool(((grid < -(1 << 20) + 1) | (grid >= (1 << 20) - 1)).any()):
            raise ValueError(f'{frame.path}: world coordinates outside memory key range')
        supervised += int((raw['label'] >= 0).sum())
    if require_labels and not supervised:
        raise ValueError(f'{manifest.split} manifest has no supervised labels')


class AugmentedFusion(nn.Module):
    """Training-only candidate perturbations; raw memory writes remain untouched."""
    def __init__(self, fusion, dropout, corruption, seed):
        super().__init__()
        for value in (dropout, corruption):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('History dropout/corruption must be probabilities in [0,1]')
        self.fusion, self.dropout, self.corruption = fusion, dropout, corruption
        self.rng = torch.Generator(device=next(fusion.parameters()).device).manual_seed(seed)

    def forward(self, features, base_logits, candidates):
        if self.training and (self.dropout or self.corruption):
            candidates = dict(candidates)
            valid = candidates['valid']
            if self.dropout:
                keep = torch.rand((len(valid), 1), device=valid.device, generator=self.rng) >= self.dropout
                candidates['valid'] = valid & keep
            if self.corruption:
                bad = (torch.rand(valid.shape, device=valid.device, generator=self.rng) < self.corruption) & valid
                # Incorrect appearance and class evidence, without labels or future frames.
                candidates['features'] = torch.where(bad[..., None], -candidates['features'], candidates['features'])
                candidates['probabilities'] = torch.where(bad[..., None],
                    candidates['probabilities'].roll(1, dims=-1), candidates['probabilities'])
        return self.fusion(features.detach(), base_logits.detach(), candidates)


class Counts:
    def __init__(self, classes):
        self.base = np.zeros((classes, classes), np.int64)
        self.fused = np.zeros_like(self.base)
        self.frames = self.points = self.history_points = 0
        self.harmed = self.rescued = self.both_correct = self.both_wrong = self.changed = 0

    def add(self, output, labels):
        labels = labels.detach().cpu().numpy()
        base = output['base_logits'].detach().argmax(-1).cpu().numpy()
        fused = output['logits'].detach().argmax(-1).cpu().numpy()
        self.frames += 1
        self.points += len(labels)
        self.history_points += int(output['has_history'].sum())
        keep = labels >= 0
        y, b, f = labels[keep], base[keep], fused[keep]
        np.add.at(self.base, (y, b), 1)
        np.add.at(self.fused, (y, f), 1)
        bc, fc = b == y, f == y
        self.harmed += int((bc & ~fc).sum())
        self.rescued += int((~bc & fc).sum())
        self.both_correct += int((bc & fc).sum())
        self.both_wrong += int((~bc & ~fc).sum())
        self.changed += int((b != f).sum())

    def report(self):
        def metrics(matrix):
            tp = matrix.diagonal()
            union = matrix.sum(0) + matrix.sum(1) - tp
            iou = np.divide(tp, union, out=np.zeros(len(tp), np.float64), where=union > 0)
            return dict(correct=int(tp.sum()), accuracy=float(tp.sum() / matrix.sum()) if matrix.sum() else None,
                        miou_fixed_classes=float(iou.mean()) if matrix.sum() else None,
                        class_iou=[float(v) if u else None for v, u in zip(iou, union)],
                        confusion=matrix.tolist())
        baseline, fused = metrics(self.base), metrics(self.fused)
        wrong = int(self.base.sum()) - baseline['correct']
        return dict(frames=self.frames, points=self.points, supervised=int(self.base.sum()),
                    history_points=self.history_points, baseline=baseline, fused=fused,
                    both_correct=self.both_correct, both_wrong=self.both_wrong, harmed=self.harmed,
                    rescued=self.rescued, net_correct=self.rescued - self.harmed, changed=self.changed,
                    harm_rate=self.harmed / baseline['correct'] if baseline['correct'] else None,
                    rescue_rate=self.rescued / wrong if wrong else None)


def _forward(model, manifest, frame, device, epoch=0):
    raw = {key: value.to(device).detach() for key, value in load_frame(manifest, frame, model.cfg.max_points).items()}
    output = model(raw['xyz'], raw['features'], raw['logits'], stamp=frame.stamp,
                   frame=frame.frame, transform=raw['transform'], epoch=epoch)
    return output, raw['label']


def evaluate(model, manifest, device='cpu'):
    if manifest.split not in ('val', 'test'):
        raise ValueError('Evaluation requires a val or test manifest')
    previous = model.training
    model.eval()
    counts = Counts(model.cfg.classes)
    try:
        with torch.no_grad():
            for sequence in manifest.sequences:
                model.reset()
                for frame in sequence.frames:
                    output, label = _forward(model, manifest, frame, device)
                    counts.add(output, label)
    finally:
        model.reset()
        model.train(previous)
    return counts.report()


def train_epoch(model, manifest, optimizer, clip_length, device, epoch):
    if manifest.split != 'train' or not set(manifest.ids) <= TRAIN_IDS:
        raise ValueError('Only permitted train sequences may reach the optimizer')
    model.train()
    total_loss = total_supervised = history_supervised = steps = 0
    for sequence in manifest.sequences:
        model.reset()
        for start in range(0, len(sequence.frames), clip_length):
            optimizer.zero_grad(set_to_none=True)
            losses, supervised, clip_history = [], 0, 0
            for frame in sequence.frames[start:start + clip_length]:
                output, label = _forward(model, manifest, frame, device, epoch)
                if not bool((label >= 0).any()):
                    continue
                loss, detail = training_loss(output, label)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite training loss')
                n = detail['supervised']
                losses.append(loss * n)
                supervised += n
                clip_history += detail['history_supervised']
                total_loss += float(loss.detach()) * n
            if clip_history:
                (torch.stack(losses).sum() / supervised).backward()
                nn.utils.clip_grad_norm_(model.fusion.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
            total_supervised += supervised
            history_supervised += clip_history
    model.reset()
    return dict(loss=total_loss / total_supervised if total_supervised else None,
                supervised=total_supervised, history_supervised=history_supervised, optimizer_steps=steps)


def _new_directory(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise ValueError(f'Refusing to overwrite output: {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _seed(seed):
    _integer(seed, 'seed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def run_training(train_path, val_path, out, *, epochs=5, clip_length=8, validate_every=1,
                 lr=1e-3, history_dropout=0.15, history_corruption=0.15, seed=0,
                 device='cpu', config=None, backbone_sha256=None):
    for name, value in (('epochs', epochs), ('clip_length', clip_length), ('validate_every', validate_every)):
        _integer(value, name, 1)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be finite and positive')
    if Path(out).exists() or Path(out).is_symlink():
        raise ValueError(f'Refusing to overwrite output: {out}')
    train, val = load_manifest(train_path, 'train'), load_manifest(val_path, 'val')
    training_pair(train, val)
    if backbone_sha256 is not None and _hash(backbone_sha256, 'backbone_sha256') != train.backbone_sha256:
        raise ValueError('Requested backbone_sha256 differs from cached backbone')
    cfg = config or MemoryConfig(feature_dim=train.feature_dim, classes=LABEL_SPACES[train.label_space])
    preflight(train, cfg, require_labels=True)
    preflight(val, cfg, require_labels=True)
    _seed(seed)
    model = CausalSemanticMemory(cfg, device)
    fusion = model.fusion
    model.fusion = AugmentedFusion(fusion, history_dropout, history_corruption, seed)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr)
    out = _new_directory(out)
    provenance = dict(scope='synthetic' if train.synthetic else 'real_training',
        deployable=not train.synthetic, backbone_sha256=train.backbone_sha256, label_space=train.label_space,
        model_config_sha256=train.model_config_sha256,
        source_fingerprint=train.source_fingerprint, feature_contract=train.feature_contract,
        manifest_sha256=dict(train=train.sha256, val=val.sha256),
        cache_fingerprint=dict(train=train.cache_fingerprint, val=val.cache_fingerprint),
        sequence_ids=dict(train=train.ids, val=val.ids), seed=seed,
        training=dict(epochs=epochs, clip_length=clip_length, validate_every=validate_every, lr=lr,
                      history_dropout=history_dropout, history_corruption=history_corruption))
    reports, steps = [], 0
    for epoch in range(1, epochs + 1):
        report = dict(epoch=epoch, train=train_epoch(model, train, optimizer, clip_length, device, epoch))
        steps += report['train']['optimizer_steps']
        if epoch % validate_every == 0 or epoch == epochs:
            report['val'] = evaluate(model, val, device)
        _save_json(out / f'epoch-{epoch:04d}.json', report)
        print(json.dumps(report, allow_nan=False), flush=True)
        reports.append(report)
    if not steps:
        raise ValueError('No supervised history reached an optimizer step; refusing a trained checkpoint')
    provenance['optimizer_steps'] = steps
    saved = checkpoint(cfg, fusion, provenance=provenance, trained=not train.synthetic)
    with (out / 'fusion.pt').open('xb') as stream:
        torch.save(saved, stream)
    result = dict(checkpoint=str(out / 'fusion.pt'), provenance=provenance, reports=reports)
    _save_json(out / 'metrics.json', result)
    return result


def evaluate_checkpoint(manifest_path, checkpoint_path, out, device='cpu'):
    if Path(out).exists() or Path(out).is_symlink():
        raise ValueError(f'Refusing to overwrite output: {out}')
    manifest = load_manifest(manifest_path)
    if manifest.split not in ('val', 'test'):
        raise ValueError('Evaluation requires a val or test manifest')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or type(saved.get('schema')) is not int or saved['schema'] != 1:
        raise ValueError('Unknown checkpoint schema')
    provenance = saved.get('provenance', {})
    for key in ('backbone_sha256', 'model_config_sha256', 'label_space', 'source_fingerprint', 'feature_contract'):
        if provenance.get(key) != getattr(manifest, key):
            raise ValueError(f'Checkpoint/cache {key} mismatch')
    if not _valid_feature_contract(provenance['feature_contract'], manifest.label_space):
        raise ValueError('Checkpoint/cache feature_contract mismatch')
    expected_scope = 'synthetic' if manifest.synthetic else 'real_training'
    if (provenance.get('scope') != expected_scope or type(saved.get('trained')) is not bool
            or saved['trained'] != (not manifest.synthetic)
            or provenance.get('deployable') is not (not manifest.synthetic)):
        raise ValueError('Checkpoint training/synthetic provenance mismatch')
    hashes = provenance.get('manifest_sha256', {})
    _keys(hashes, ('train', 'val'), context='checkpoint manifest_sha256')
    for key, value in hashes.items():
        _hash(value, f'{key} manifest hash')
    if set(manifest.ids) & set(provenance.get('sequence_ids', {}).get('train', [])):
        raise ValueError('Evaluation sequences overlap checkpoint training')
    cfg = MemoryConfig(**saved['config'])
    preflight(manifest, cfg)
    _seed(0)
    model = CausalSemanticMemory(cfg, device)
    model.fusion.load_state_dict(saved['fusion'], strict=True)
    metrics = evaluate(model, manifest, device)
    result = dict(split=manifest.split, manifest_sha256=manifest.sha256,
                  cache_fingerprint=manifest.cache_fingerprint, checkpoint_sha256=_digest(checkpoint_path),
                  model_config_sha256=manifest.model_config_sha256,
                  feature_contract=manifest.feature_contract, label_space=manifest.label_space,
                  scope=expected_scope, deployable=provenance['deployable'], metrics=metrics)
    out = _new_directory(out)
    _save_json(out / 'metrics.json', result)
    return result


def smoke(out):
    """Tiny generated caches; no dataset, backbone, download, or deployable output."""
    out = _new_directory(out)
    metadata = dict(schema=1, label_space='native16', feature_dim=64,
        backbone_sha256=hashlib.sha256(b'synthetic backbone, not weights').hexdigest(),
        model_config_sha256=hashlib.sha256(b'synthetic model configuration, not a deployed config').hexdigest(),
        source_fingerprint=hashlib.sha256(b'semantic-memory synthetic cache v1').hexdigest(),
        feature_contract=feature_contract('native16'), synthetic=True)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(8, 64)).astype(np.float32)
    for split, sid, offset in (('train', '00', 0.), ('val', '08', 0.1), ('test', '07', 0.2)):
        directory = out / split
        directory.mkdir()
        frames = []
        for frame in range(4):
            xyz = np.column_stack((np.arange(8) + 5. + offset, np.ones(8), np.zeros(8))).astype(np.float32)
            label = np.arange(8, dtype=np.int64) % 2
            logits = np.zeros((8, 16), np.float32)
            logits[np.arange(8), label] = 0.5
            if frame % 2:
                logits[np.arange(4), 1 - label[:4]] = 0.7
            name = f'{frame:06d}.npz'
            np.savez(directory / name, xyz=xyz, features=features.astype(np.float16), logits=logits,
                     transform=np.eye(4, dtype=np.float32), label=label,
                     metadata=np.array(json.dumps(metadata)))
            frames.append(dict(frame=frame, stamp=frame * 0.1, path=name))
        _save_json(directory / 'manifest.json', dict(metadata, split=split, sequences=[dict(id=sid, frames=frames)]))
    cfg = MemoryConfig(capacity=64, neighbors=2, hidden_dim=8, chunk_size=16, max_points=64)
    result = run_training(out / 'train/manifest.json', out / 'val/manifest.json', out / 'run',
                          epochs=2, clip_length=2, config=cfg, seed=0)
    result['test'] = evaluate_checkpoint(out / 'test/manifest.json', out / 'run/fusion.pt', out / 'scores')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Train fusion only on validated chronological cached clips')
    train.add_argument('--train', required=True)
    train.add_argument('--val', required=True)
    train.add_argument('--out', required=True, help='New output directory; existing paths are refused')
    train.add_argument('--epochs', type=int, default=5)
    train.add_argument('--clip-length', type=int, default=8)
    train.add_argument('--validate-every', type=int, default=1)
    train.add_argument('--lr', type=float, default=1e-3)
    train.add_argument('--history-dropout', type=float, default=0.15)
    train.add_argument('--history-corruption', type=float, default=0.15)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--device', default='cpu')
    train.add_argument('--backbone-sha256', help='Optional trusted backbone digest to pin against cache metadata')
    train.add_argument('--chunk-size', type=int, default=MemoryConfig().chunk_size)
    train.add_argument('--capacity', type=int, default=MemoryConfig().capacity)
    evaluation = commands.add_parser('evaluate', help='Evaluate normal history on val or test; never optimize')
    evaluation.add_argument('--manifest', required=True)
    evaluation.add_argument('--checkpoint', required=True)
    evaluation.add_argument('--out', required=True)
    evaluation.add_argument('--device', default='cpu')
    synthetic = commands.add_parser('smoke', help='Generate tiny synthetic caches and a NON-deployable checkpoint')
    synthetic.add_argument('--out', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    try:
        if command == 'train':
            train_path, val_path = args.pop('train'), args.pop('val')
            manifest = load_manifest(train_path, 'train')
            cfg = MemoryConfig(classes=LABEL_SPACES[manifest.label_space], chunk_size=args.pop('chunk_size'),
                               capacity=args.pop('capacity'))
            result = run_training(train_path, val_path, config=cfg, **args)
        elif command == 'evaluate':
            result = evaluate_checkpoint(args.pop('manifest'), args.pop('checkpoint'), **args)
        else:
            result = smoke(**args)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
    if command == 'evaluate':
        print(json.dumps(result, allow_nan=False))
    else:
        print(json.dumps(dict(checkpoint=result['checkpoint'], scope=result['provenance']['scope'],
                              deployable=result['provenance']['deployable']), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
