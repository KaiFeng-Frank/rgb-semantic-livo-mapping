"""Synthetic-only guards for cached causal fusion. No backbone or dataset needed.

Run: /home/frank/semantic_livo_runtime/venv/bin/python -m pytest tools/semantic_memory/test_training.py
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.semantic_memory import train as T


@pytest.fixture(scope='module', autouse=True)
def small_cpu_workload():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def make_cache(root, split='train', ids=None, frames=4, space='native16', labels=True, published=False):
    """Every generated observation, manifest and trained checkpoint is synthetic."""
    root = Path(root)
    root.mkdir(parents=True)
    ids = ids or [{'train': '00', 'val': '08', 'test': '07'}[split]]
    metadata = dict(schema=1, feature_dim=64, label_space=space,
                    backbone_sha256='a' * 64, model_config_sha256='d' * 64, source_fingerprint='b' * 64,
                    feature_contract=T.feature_contract(space), synthetic=True)
    sequences = []
    rng = np.random.default_rng(2)
    features = rng.normal(size=(6, 64)).astype(np.float32)
    for sid in ids:
        entries = []
        for frame in range(frames):
            name = f'{sid}-{frame:06d}.npz'
            xyz = np.column_stack((np.arange(6) + 5. + int(sid) * .01,
                                   np.ones(6), np.zeros(6))).astype(np.float32)
            label = np.arange(6, dtype=np.int64) % 2
            logits = np.full((6, T.LABEL_SPACES[space]), -1., np.float32)
            logits[np.arange(6), label] = .3
            if frame % 2:
                logits[np.arange(3), 1 - label[:3]] = .4
            arrays = dict(xyz=xyz, features=features.astype(np.float16), logits=logits,
                          transform=np.eye(4, dtype=np.float32), metadata=np.array(json.dumps(metadata)))
            if labels:
                arrays['label'] = label
            np.savez(root / name, **arrays)
            entry = dict(frame=frame, stamp=100. + frame * .1, path=name)
            if published:
                entry['sha256'] = digest(root / name)
            entries.append(entry)
        sequences.append(dict(id=sid, frames=entries))
    path = root / 'manifest.json'
    write_json(path, dict(metadata, split=split, sequences=sequences))
    return path


def modify_manifest(path, callback):
    data = json.loads(Path(path).read_text())
    callback(data)
    write_json(path, data)


def modify_npz(path, callback):
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    callback(arrays)
    np.savez(path, **arrays)


def tiny_config(space='native16', **kwargs):
    return T.MemoryConfig(classes=T.LABEL_SPACES[space], capacity=32, neighbors=2,
                          hidden_dim=8, chunk_size=3, max_points=32, **kwargs)


def test_manifest_relative_paths_optional_published_hash_and_float_cache(tmp_path):
    path = make_cache(tmp_path / 'cache', published=True)
    manifest = T.load_manifest(path, 'train')
    assert manifest.sha256 == digest(path)
    assert manifest.ids == ['00']
    assert manifest.model_config_sha256 == manifest.metadata['model_config_sha256'] == 'd' * 64
    assert manifest.metadata['feature_contract']['feature_source'] == 'unfused_decoder'
    assert manifest.feature_contract['half'] is True
    assert manifest.feature_contract['shuffle'] is False
    raw = T.load_frame(manifest, manifest.frames[0])
    assert raw['features'].dtype == torch.float16
    assert not any(value.requires_grad for value in raw.values())
    T.preflight(manifest, tiny_config(), require_labels=True)


@pytest.mark.parametrize('sid', ['03', '07', '08', '09', '11', '7', '007', 7])
def test_forbidden_and_noncanonical_train_ids(tmp_path, sid):
    path = make_cache(tmp_path / 'cache')
    modify_manifest(path, lambda d: d['sequences'][0].update(id=sid))
    with pytest.raises(ValueError, match='forbidden'):
        T.load_manifest(path)


def test_all_and_only_permitted_train_sequences(tmp_path):
    manifest = T.load_manifest(make_cache(tmp_path / 'cache', ids=sorted(T.TRAIN_IDS), frames=1))
    assert set(manifest.ids) == {'00', '01', '02', '04', '05', '06', '10'}


@pytest.mark.parametrize('split,sid', [('val', '07'), ('val', '09'), ('val', '00'), ('test', '08')])
def test_validation_cannot_tune_legacy_test_sequences(tmp_path, split, sid):
    path = make_cache(tmp_path / 'cache', split=split)
    modify_manifest(path, lambda d: d['sequences'][0].update(id=sid))
    with pytest.raises(ValueError, match='forbidden'):
        T.load_manifest(path)


@pytest.mark.parametrize('key,value,match', [
    ('schema', 2, 'schema'), ('schema', True, 'schema'), ('feature_dim', 32, 'feature_dim'),
    ('label_space', 'common9', 'label_space'), ('backbone_sha256', 'guess', 'SHA-256'),
    ('model_config_sha256', 'guess', 'SHA-256'), ('model_config_sha256', None, 'SHA-256'),
    ('source_fingerprint', 'c' * 63, 'SHA-256'), ('synthetic', 'true', 'boolean'),
    ('feature_contract', {}, 'feature_contract'), ('split', 'training', 'split'),
])
def test_manifest_header_is_strict(tmp_path, key, value, match):
    path = make_cache(tmp_path / 'cache')
    modify_manifest(path, lambda d: d.update({key: value}))
    with pytest.raises(ValueError, match=match):
        T.load_manifest(path)


@pytest.mark.parametrize('key,value', [('grid_size', .1), ('intensity_scale', 1.),
                                      ('feature_source', 'fused_decoder'), ('geometry', 'world'),
                                      ('label_space', 'semantic-kitti19'), ('half', False), ('half', 1),
                                      ('shuffle', True), ('shuffle', 0)])
def test_feature_contract_is_exact(tmp_path, key, value):
    path = make_cache(tmp_path / 'cache')
    modify_manifest(path, lambda d: d['feature_contract'].update({key: value}))
    with pytest.raises(ValueError, match='feature_contract'):
        T.load_manifest(path)


@pytest.mark.parametrize('location', ['manifest', 'npz'])
def test_model_config_hash_is_mandatory(tmp_path, location):
    path = make_cache(tmp_path / 'cache')
    if location == 'manifest':
        modify_manifest(path, lambda d: d.pop('model_config_sha256'))
        with pytest.raises(ValueError, match='missing=.*model_config_sha256'):
            T.load_manifest(path)
    else:
        def missing(z):
            metadata = json.loads(z['metadata'].item())
            metadata.pop('model_config_sha256')
            z['metadata'] = np.array(json.dumps(metadata))
        modify_npz(path.parent / '00-000000.npz', missing)
        manifest = T.load_manifest(path)
        with pytest.raises(ValueError, match='missing=.*model_config_sha256'):
            T.load_frame(manifest, manifest.frames[0])


@pytest.mark.parametrize('key,value', [('frame', 0), ('frame', -1), ('frame', True),
                                      ('stamp', 100.), ('stamp', 99.), ('stamp', True)])
def test_rejects_duplicate_backward_or_invalid_frame_and_time(tmp_path, key, value):
    path = make_cache(tmp_path / 'cache')
    modify_manifest(path, lambda d: d['sequences'][0]['frames'][1].update({key: value}))
    with pytest.raises(ValueError):
        T.load_manifest(path)


def test_nonfinite_json_and_duplicate_keys_are_not_accepted(tmp_path):
    path = make_cache(tmp_path / 'cache')
    modify_manifest(path, lambda d: d['sequences'][0]['frames'][0].update(stamp=float('nan')))
    with pytest.raises(ValueError, match='Nonfinite'):
        T.load_manifest(path)
    path.write_text('{"schema":1,"schema":1}')
    with pytest.raises(ValueError, match='Duplicate JSON'):
        T.load_manifest(path)


def test_duplicate_sequences_and_paths_rejected(tmp_path):
    path = make_cache(tmp_path / 'cache')
    original = json.loads(path.read_text())
    modify_manifest(path, lambda d: d['sequences'].append(copy.deepcopy(d['sequences'][0])))
    with pytest.raises(ValueError, match='Duplicate sequence'):
        T.load_manifest(path)
    write_json(path, original)
    modify_manifest(path, lambda d: d['sequences'][0]['frames'][1].update(path=d['sequences'][0]['frames'][0]['path']))
    with pytest.raises(ValueError, match='Duplicate cache'):
        T.load_manifest(path)


@pytest.mark.parametrize('kind', ['absolute', 'escape', 'symlink'])
def test_paths_must_be_relative_and_contained(tmp_path, kind):
    path = make_cache(tmp_path / 'cache')
    outside = tmp_path / 'outside.npz'
    outside.write_bytes((path.parent / '00-000000.npz').read_bytes())
    if kind == 'symlink':
        (path.parent / 'link.npz').symlink_to(outside)
    name = {'absolute': str(outside), 'escape': '../outside.npz', 'symlink': 'link.npz'}[kind]
    modify_manifest(path, lambda d: d['sequences'][0]['frames'][0].update(path=name))
    with pytest.raises(ValueError, match='relative|contained'):
        T.load_manifest(path)


def test_published_hash_and_post_preflight_tamper_rejected(tmp_path):
    path = make_cache(tmp_path / 'cache', published=True)
    manifest = T.load_manifest(path)
    modify_npz(manifest.frames[0].path, lambda z: z['features'].__setitem__((0, 0), .125))
    with pytest.raises(ValueError, match='changed'):
        T.load_frame(manifest, manifest.frames[0])
    with pytest.raises(ValueError, match='published frame sha256 mismatch'):
        T.load_manifest(path)


@pytest.mark.parametrize('key,value', [('schema', True), ('schema', 2), ('feature_dim', 32),
                                      ('backbone_sha256', 'c' * 64), ('source_fingerprint', 'c' * 64),
                                      ('model_config_sha256', 'c' * 64),
                                      ('label_space', 'semantic-kitti19'), ('synthetic', False)])
def test_per_npz_metadata_must_match_manifest(tmp_path, key, value):
    path = make_cache(tmp_path / 'cache')
    def corrupt(z):
        metadata = json.loads(z['metadata'].item())
        metadata[key] = value
        z['metadata'] = np.array(json.dumps(metadata))
    modify_npz(path.parent / '00-000000.npz', corrupt)
    manifest = T.load_manifest(path)
    with pytest.raises(ValueError, match='metadata'):
        T.load_frame(manifest, manifest.frames[0])


@pytest.mark.parametrize('key,value', [('half', False), ('half', 1), ('shuffle', True), ('shuffle', 0)])
def test_npz_half_and_shuffle_match_strict_boolean_contract(tmp_path, key, value):
    path = make_cache(tmp_path / 'cache')
    def corrupt(z):
        metadata = json.loads(z['metadata'].item())
        metadata['feature_contract'][key] = value
        z['metadata'] = np.array(json.dumps(metadata))
    modify_npz(path.parent / '00-000000.npz', corrupt)
    manifest = T.load_manifest(path)
    with pytest.raises(ValueError, match='feature_contract mismatch'):
        T.load_frame(manifest, manifest.frames[0])


@pytest.mark.parametrize('mutation,match', [
    (lambda z: z.update(features=np.zeros((6, 63), np.float32)), 'features'),
    (lambda z: z.update(features=z['features'].astype(np.float64)), 'float16 or float32'),
    (lambda z: z.update(features=np.full((6, 64), np.nan, np.float32)), 'nonfinite'),
    (lambda z: z.update(features=np.full((6, 64), 65505., np.float32)), 'float16 memory storage range'),
    (lambda z: z.update(features=np.full((6, 64), 'bad', dtype=object)), 'Object arrays'),
    (lambda z: z.update(logits=np.zeros((6, 19), np.float32)), 'logits'),
    (lambda z: z.update(xyz=np.zeros((6, 4), np.float32)), 'xyz'),
    (lambda z: z.update(label=np.full(6, 16, np.int32)), 'label outside'),
    (lambda z: z.update(label=np.full(6, -2, np.int32)), 'label outside'),
    (lambda z: z.update(label=np.zeros((6, 1), np.int32)), 'label'),
    (lambda z: z.update(label=np.zeros(6, np.float32)), 'integer'),
    (lambda z: z.update(fused_features=z['features']), 'unexpected'),
    (lambda z: z.pop('metadata'), 'missing'),
    (lambda z: z.update(metadata=np.array([z['metadata'].item()])), 'scalar Unicode'),
    (lambda z: z['transform'].__setitem__((0, 0), -1), 'SE\\(3\\)'),
    (lambda z: z['transform'].__setitem__((3, 0), 1), 'SE\\(3\\)'),
])
def test_array_shape_dtype_labels_geometry_and_raw_only_guards(tmp_path, mutation, match):
    path = make_cache(tmp_path / 'cache')
    modify_npz(path.parent / '00-000000.npz', mutation)
    manifest = T.load_manifest(path)
    with pytest.raises(ValueError, match=match):
        T.load_frame(manifest, manifest.frames[0])


def test_optional_labels_empty_frames_and_explicit_19_class_space(tmp_path):
    path = make_cache(tmp_path / 'cache', space='semantic-kitti19', labels=False)
    manifest = T.load_manifest(path)
    assert T.load_frame(manifest, manifest.frames[0])['label'].tolist() == [-1] * 6
    T.preflight(manifest, tiny_config('semantic-kitti19'))
    with pytest.raises(ValueError, match='class dimensions'):
        T.preflight(manifest, tiny_config())
    with pytest.raises(ValueError, match='no supervised'):
        T.preflight(manifest, tiny_config('semantic-kitti19'), require_labels=True)
    def empty(z):
        for key in ('xyz', 'features', 'logits'):
            z[key] = z[key][:0]
    modify_npz(path.parent / '00-000000.npz', empty)
    manifest = T.load_manifest(path)
    assert T.load_frame(manifest, manifest.frames[0])['label'].numel() == 0


@pytest.mark.parametrize('key,value', [('backbone_sha256', 'c' * 64), ('source_fingerprint', 'c' * 64),
                                      ('model_config_sha256', 'c' * 64),
                                      ('label_space', 'semantic-kitti19'), ('synthetic', False)])
def test_train_val_must_share_backbone_source_and_ontology(tmp_path, key, value):
    train_path = make_cache(tmp_path / 'train')
    val_path = make_cache(tmp_path / 'val', 'val')
    def change(d):
        d[key] = value
        if key == 'label_space':
            d['feature_contract'] = T.feature_contract(value)
    modify_manifest(val_path, change)
    with pytest.raises(ValueError, match=key):
        T.training_pair(T.load_manifest(train_path), T.load_manifest(val_path))


def test_cross_split_copied_content_rejected(tmp_path):
    train_path = make_cache(tmp_path / 'train')
    val_path = make_cache(tmp_path / 'val', 'val')
    (val_path.parent / '08-000000.npz').write_bytes((train_path.parent / '00-000000.npz').read_bytes())
    with pytest.raises(ValueError, match='content must be disjoint'):
        T.training_pair(T.load_manifest(train_path), T.load_manifest(val_path))


def test_test_split_never_reaches_training_or_selection(tmp_path, monkeypatch):
    train_path = make_cache(tmp_path / 'train')
    val_path = make_cache(tmp_path / 'val', 'val')
    test_path = make_cache(tmp_path / 'test', 'test')
    monkeypatch.setattr(T.torch.optim, 'Adam', lambda *a, **kw: pytest.fail('optimizer constructed'))
    for train, val in ((test_path, val_path), (train_path, test_path)):
        with pytest.raises(ValueError, match='Expected split'):
            T.run_training(train, val, tmp_path / 'out')
    with pytest.raises(ValueError, match='test training is forbidden'):
        T.training_pair(T.load_manifest(test_path), T.load_manifest(val_path))
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('bad_feature', [np.nan, 65505.])
def test_all_caches_preflighted_before_optimizer_or_output(tmp_path, monkeypatch, bad_feature):
    train_path = make_cache(tmp_path / 'train')
    val_path = make_cache(tmp_path / 'val', 'val')
    modify_npz(val_path.parent / '08-000003.npz',
               lambda z: z.update(features=np.full((6, 64), bad_feature, np.float32)))
    monkeypatch.setattr(T.torch.optim, 'Adam', lambda *a, **kw: pytest.fail('optimizer constructed'))
    with pytest.raises(ValueError, match='nonfinite|float16 memory storage range'):
        T.run_training(train_path, val_path, tmp_path / 'out', config=tiny_config())
    assert not (tmp_path / 'out').exists()


def test_point_capacity_and_coordinate_range_guard(tmp_path):
    path = make_cache(tmp_path / 'cache')
    manifest = T.load_manifest(path)
    with pytest.raises(ValueError, match='max_points'):
        T.load_frame(manifest, manifest.frames[0], max_points=5)
    modify_npz(path.parent / '00-000000.npz', lambda z: z['transform'].__setitem__((0, 3), 1e7))
    with pytest.raises(ValueError, match='memory key range'):
        T.preflight(T.load_manifest(path), tiny_config())


def test_harm_and_rescue_counts_not_hidden_by_equal_accuracy():
    def logits(labels):
        return torch.nn.functional.one_hot(torch.tensor(labels), 4).float()
    counts = T.Counts(4)
    counts.add(dict(base_logits=logits([0, 0, 2, 0, 0]), logits=logits([0, 1, 0, 1, 3]),
                    has_history=torch.tensor([False, True, True, True, True])), torch.tensor([0, 1, 2, 3, -1]))
    result = counts.report()
    assert result['supervised'] == 4
    assert result['baseline']['accuracy'] == result['fused']['accuracy'] == .5
    assert result['harmed'] == result['rescued'] == result['both_correct'] == result['both_wrong'] == 1
    assert result['harm_rate'] == result['rescue_rate'] == .5
    assert result['net_correct'] == 0 and result['changed'] == 3


def test_augmentation_is_training_only_and_never_rewrites_raw_memory(tmp_path):
    manifest = T.load_manifest(make_cache(tmp_path / 'cache'))
    model = T.CausalSemanticMemory(tiny_config())
    model.fusion = T.AugmentedFusion(model.fusion, 1., 1., 4)
    model.train()
    T._forward(model, manifest, manifest.frames[0], 'cpu')
    output, _ = T._forward(model, manifest, manifest.frames[1], 'cpu')
    assert not output['has_history'].any()
    torch.testing.assert_close(output['logits'], output['base_logits'])
    raw = T.load_frame(manifest, manifest.frames[1])
    # One representative per spatial cell, newest records first within each cell.
    latest = model.memory.frame == 1
    torch.testing.assert_close(model.memory.features[latest], raw['features'])
    torch.testing.assert_close(model.memory.probabilities[latest], raw['logits'].softmax(-1).half())
    model.reset()
    model.eval()
    state = model.fusion.rng.get_state().clone()
    T._forward(model, manifest, manifest.frames[0], 'cpu')
    output, _ = T._forward(model, manifest, manifest.frames[1], 'cpu')
    assert output['has_history'].all()
    torch.testing.assert_close(state, model.fusion.rng.get_state())


def test_corruption_and_dropout_deterministic_without_modifying_candidates(tmp_path):
    manifest = T.load_manifest(make_cache(tmp_path / 'cache'))
    torch.manual_seed(1)
    model = T.CausalSemanticMemory(tiny_config())
    raw = T.load_frame(manifest, manifest.frames[0])
    T._forward(model, manifest, manifest.frames[0], 'cpu')
    candidates = model.memory.retrieve(raw['xyz'], raw['features'], manifest.frames[1].stamp)
    original = {key: value.clone() for key, value in candidates.items()}
    first = T.AugmentedFusion(copy.deepcopy(model.fusion), .5, 1., 11)
    second = T.AugmentedFusion(copy.deepcopy(model.fusion), .5, 1., 11)
    a = first(raw['features'], raw['logits'], candidates)
    b = second(raw['features'], raw['logits'], candidates)
    for key in a:
        torch.testing.assert_close(a[key], b[key])
    for key in original:
        torch.testing.assert_close(candidates[key], original[key])


@pytest.fixture
def trained(tmp_path):
    train_path = make_cache(tmp_path / 'train')
    val_path = make_cache(tmp_path / 'val', 'val')
    result = T.run_training(train_path, val_path, tmp_path / 'run', epochs=2,
                            clip_length=2, config=tiny_config(), seed=17,
                            history_dropout=0., history_corruption=0.)
    return train_path, val_path, result


def test_training_changes_fusion_with_gradients_chronological_clips_and_resets(tmp_path, monkeypatch):
    train_path = make_cache(tmp_path / 'train', ids=['00', '01'])
    val_path = make_cache(tmp_path / 'val', 'val')
    before = {p: digest(p) for p in tmp_path.rglob('*.npz')}
    records = []
    class TrackingMemory(T.CausalSemanticMemory):
        def forward(self, xyz, features, logits, **kwargs):
            records.append((self.training, kwargs['epoch'], kwargs['frame'], self.memory.last_frame,
                            torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                            features.requires_grad, logits.requires_grad))
            return super().forward(xyz, features, logits, **kwargs)
    monkeypatch.setattr(T, 'CausalSemanticMemory', TrackingMemory)
    cfg = tiny_config()
    T._seed(9)
    initial = T.CausalSemanticMemory(cfg).fusion.state_dict()
    result = T.run_training(train_path, val_path, tmp_path / 'run', epochs=2, clip_length=2,
                            validate_every=2, config=cfg, seed=9, history_dropout=0., history_corruption=0.)
    training = [row for row in records if row[0]]
    assert [(r[1], r[2]) for r in training] == [(e, f) for e in (1, 2) for _ in range(2) for f in range(4)]
    assert all(r[3] == (None if r[2] == 0 else r[2] - 1) for r in training)
    assert all(r[4] and not any(r[5:]) for r in training)
    assert all(not r[4] for r in records if not r[0])
    assert 'val' not in result['reports'][0] and 'val' in result['reports'][1]
    assert result['provenance']['optimizer_steps'] == 8
    saved = torch.load(result['checkpoint'], weights_only=True)
    assert any(not torch.equal(initial[k], saved['fusion'][k]) for k in initial)
    T.CausalSemanticMemory(cfg).fusion.load_state_dict(saved['fusion'], strict=True)
    assert before == {p: digest(p) for p in tmp_path.rglob('*.npz')}


def test_synthetic_checkpoint_contract_and_manifest_source_provenance(trained):
    train_path, val_path, result = trained
    saved = torch.load(result['checkpoint'], weights_only=True)
    assert saved['trained'] is False
    provenance = saved['provenance']
    assert provenance['scope'] == 'synthetic' and provenance['deployable'] is False
    assert provenance['manifest_sha256'] == dict(train=digest(train_path), val=digest(val_path))
    assert provenance['feature_contract'] == T.feature_contract('native16')
    assert provenance['source_fingerprint'] == 'b' * 64
    assert provenance['backbone_sha256'] == 'a' * 64
    assert provenance['model_config_sha256'] == 'd' * 64
    assert provenance['sequence_ids'] == dict(train=['00'], val=['08'])
    assert provenance['cache_fingerprint']['train'] == T.load_manifest(train_path).cache_fingerprint
    assert {'harmed', 'rescued', 'baseline', 'fused'} <= result['reports'][-1]['val'].keys()


def test_repeat_training_seed_and_stable_normal_history_validation(trained, tmp_path):
    train_path, val_path, result = trained
    repeated = T.run_training(train_path, val_path, tmp_path / 'repeat', epochs=2,
                              clip_length=2, config=tiny_config(), seed=17,
                              history_dropout=0., history_corruption=0.)
    first = torch.load(result['checkpoint'], weights_only=True)
    second = torch.load(repeated['checkpoint'], weights_only=True)
    assert result['reports'] == repeated['reports']
    for key in first['fusion']:
        torch.testing.assert_close(first['fusion'][key], second['fusion'][key], rtol=0, atol=0)
    manifest = T.load_manifest(val_path)
    model = T.CausalSemanticMemory(tiny_config())
    model.fusion.load_state_dict(first['fusion'])
    model.fusion = T.AugmentedFusion(model.fusion, 1., 1., 2)
    model.train()
    state = model.fusion.rng.get_state().clone()
    a, b = T.evaluate(model, manifest), T.evaluate(model, manifest)
    assert a == b and a['history_points'] > 0 and model.training
    torch.testing.assert_close(state, model.fusion.rng.get_state())
    assert model.memory.last_frame is None
    for key, value in model.fusion.fusion.state_dict().items():
        torch.testing.assert_close(value, first['fusion'][key], rtol=0, atol=0)


def test_evaluation_checkpoint_native_ontology_and_test_only_read(trained, tmp_path):
    _, _, result = trained
    path = make_cache(tmp_path / 'test', 'test')
    a = T.evaluate_checkpoint(path, result['checkpoint'], tmp_path / 'score-a')
    b = T.evaluate_checkpoint(path, result['checkpoint'], tmp_path / 'score-b')
    assert a == b and a['split'] == 'test'
    assert a['metrics']['supervised'] == 24 and a['deployable'] is False
    assert a['checkpoint_sha256'] == digest(result['checkpoint'])
    assert a['model_config_sha256'] == 'd' * 64
    path19 = make_cache(tmp_path / 'test19', 'test', space='semantic-kitti19')
    with pytest.raises(ValueError, match='label_space'):
        T.evaluate_checkpoint(path19, result['checkpoint'], tmp_path / 'bad')


def test_explicit_19_class_training_and_offline_evaluation_stay_synthetic(tmp_path):
    train_path = make_cache(tmp_path / 'train19', space='semantic-kitti19')
    val_path = make_cache(tmp_path / 'val19', 'val', space='semantic-kitti19')
    # Exercise the final native class, with no mapping from a 16-class ontology.
    def final_class(z):
        z['label'][0] = 18
        z['logits'][0] = -1.
        z['logits'][0, 18] = 1.
    for path in tmp_path.rglob('*.npz'):
        modify_npz(path, final_class)
    result = T.run_training(train_path, val_path, tmp_path / 'run19', epochs=1,
                            clip_length=2, config=tiny_config('semantic-kitti19'),
                            history_dropout=0., history_corruption=0.)
    saved = torch.load(result['checkpoint'], weights_only=True)
    assert saved['config']['classes'] == 19 and saved['provenance']['label_space'] == 'semantic-kitti19'
    assert saved['trained'] is False and saved['provenance']['deployable'] is False
    evaluated = T.evaluate_checkpoint(val_path, result['checkpoint'], tmp_path / 'score19')
    assert len(evaluated['metrics']['baseline']['confusion']) == 19
    assert sum(evaluated['metrics']['baseline']['confusion'][18]) == 4


@pytest.mark.parametrize('key,value', [('backbone_sha256', 'c' * 64), ('source_fingerprint', 'c' * 64),
                                      ('model_config_sha256', 'c' * 64),
                                      ('scope', 'real_training'), ('deployable', True)])
def test_evaluation_refuses_checkpoint_provenance_mismatch(trained, tmp_path, key, value):
    _, val, result = trained
    saved = torch.load(result['checkpoint'], weights_only=True)
    saved['provenance'][key] = value
    tampered = tmp_path / 'tampered.pt'
    torch.save(saved, tampered)
    with pytest.raises(ValueError, match='mismatch'):
        T.evaluate_checkpoint(val, tampered, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_evaluation_requires_checkpoint_model_config_hash(trained, tmp_path):
    _, val, result = trained
    saved = torch.load(result['checkpoint'], weights_only=True)
    saved['provenance'].pop('model_config_sha256')
    tampered = tmp_path / 'missing-config.pt'
    torch.save(saved, tampered)
    with pytest.raises(ValueError, match='model_config_sha256 mismatch'):
        T.evaluate_checkpoint(val, tampered, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('key,value', [('half', False), ('half', 1), ('shuffle', True), ('shuffle', 0)])
def test_evaluation_requires_checkpoint_half_and_shuffle_contract(trained, tmp_path, key, value):
    _, val, result = trained
    saved = torch.load(result['checkpoint'], weights_only=True)
    saved['provenance']['feature_contract'][key] = value
    tampered = tmp_path / 'wrong-contract.pt'
    torch.save(saved, tampered)
    with pytest.raises(ValueError, match='feature_contract mismatch'):
        T.evaluate_checkpoint(val, tampered, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_no_labels_or_no_supervised_history_cannot_emit_trained_checkpoint(tmp_path):
    train_path = make_cache(tmp_path / 'unlabeled', labels=False)
    val_path = make_cache(tmp_path / 'val', 'val')
    with pytest.raises(ValueError, match='no supervised labels'):
        T.run_training(train_path, val_path, tmp_path / 'none', config=tiny_config())
    assert not (tmp_path / 'none').exists()
    single = make_cache(tmp_path / 'single', frames=1)
    with pytest.raises(ValueError, match='No supervised history'):
        T.run_training(single, val_path, tmp_path / 'no-history', epochs=1, config=tiny_config())
    assert not (tmp_path / 'no-history/fusion.pt').exists()


def test_existing_outputs_refused_without_changes(trained, tmp_path):
    train_path, val_path, result = trained
    out = Path(result['checkpoint']).parent
    before = {p: digest(p) for p in out.iterdir()}
    with pytest.raises(ValueError, match='overwrite'):
        T.run_training(train_path, val_path, out)
    with pytest.raises(ValueError, match='overwrite'):
        T.evaluate_checkpoint(val_path, result['checkpoint'], out)
    assert before == {p: digest(p) for p in out.iterdir()}
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='overwrite'):
        T.smoke(empty)


def test_cli_help_and_synthetic_smoke(tmp_path):
    script = Path(T.__file__)
    help_result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert help_result.returncode == 0 and 'source_fingerprint' in help_result.stdout
    # In-process main keeps this tiny CPU workload from spawning a large BLAS pool.
    assert T.main(['smoke', '--out', str(tmp_path / 'smoke')]) == 0
    saved = torch.load(tmp_path / 'smoke/run/fusion.pt', weights_only=True)
    assert saved['trained'] is False and saved['provenance']['scope'] == 'synthetic'
    assert saved['provenance']['deployable'] is False
    manifest = T.load_manifest(tmp_path / 'smoke/train/manifest.json')
    assert saved['provenance']['model_config_sha256'] == manifest.model_config_sha256
    scores = json.loads((tmp_path / 'smoke/scores/metrics.json').read_text())
    assert scores['metrics']['history_points'] > 0
    assert T.main(['evaluate', '--manifest', str(tmp_path / 'smoke/test/manifest.json'),
                   '--checkpoint', str(tmp_path / 'smoke/run/fusion.pt'), '--out', str(tmp_path / 'cli-score')]) == 0
    with pytest.raises(SystemExit) as error:
        T.main(['train', '--train', str(tmp_path / 'smoke/test/manifest.json'),
                '--val', str(tmp_path / 'smoke/val/manifest.json'), '--out', str(tmp_path / 'forbidden')])
    assert error.value.code == 2
