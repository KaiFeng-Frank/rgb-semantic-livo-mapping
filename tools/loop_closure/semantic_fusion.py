"""Bounded, reversible semantic voxel fusion; only NumPy and the standard library.

Points must already be deskewed in the same body frame as T_map_body. Each
accepted stamp owns an immutable copy of its eligible local observations, and
one replaceable contribution per spatial tile. Pose revisions replace those
contributions, then reduce ONLY the union of old/new tiles. They never subtract
floating point totals or reproject unrelated raw history. Moving every pose
necessarily reprojects every retained observation.

Votes are confidence-weighted point counts (not calibrated probabilities).
Snapshots use voxel centers, lowest-class-ID tie breaking, and winner/total
vote share as confidence. Zero-confidence, excluded, and below-threshold points
have no persistent contribution. The point cap applies BEFORE filtering; the
session cap counts the filtered observations actually retained in memory.

Mutations are transactional for validation/geometry failures. This object is
single-writer: callers must serialize access, including snapshots/diagnostics.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np


# src/ptv3_loader_verified.py checkpoint order: barrier=0, cone=7,
# driveable_surface=10. The caller may override this policy, including with ().
DEFAULT_EXCLUDED_CLASSES = (1, 2, 3, 4, 5, 6, 8, 9)
_I64_MAX = np.iinfo(np.int64).max


def _integer(value, name, minimum=1, maximum=_I64_MAX):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= int(value) <= maximum):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return int(value)


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must contain real numbers')
    return array


def _readonly(array):
    array.setflags(write=False)
    return array


def _poses(value, count):
    raw = _real_array(value, 'poses')
    if count == 0 and raw.shape == (0,):
        raw = raw.reshape(0, 4, 4)
    if raw.shape != (count, 4, 4):
        raise ValueError('poses must have shape (len(times_ns), 4, 4)')
    poses = np.array(raw, dtype=np.float64, copy=True)
    rotations = poses[:, :3, :3]
    if (not np.isfinite(poses).all()
            or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(rotations.transpose(0, 2, 1) @ rotations,
                               np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0)):
        raise ValueError('poses must be finite rigid T_map_body transforms')
    return _readonly(poses)


@dataclass(frozen=True)
class _Config:
    voxel: float
    classes: int
    tile_voxels: int
    max_keyframes: int
    max_points: int
    excluded_classes: tuple
    minimum_confidence: float
    max_total_points: int


@dataclass(frozen=True)
class _Packet:
    points: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    input_points: int
    excluded_points: int
    low_confidence_points: int


@dataclass(frozen=True)
class _Contribution:
    # Sorted sparse (local voxel * classes + label), without a dense class axis.
    codes: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    voxels: np.ndarray


@dataclass(frozen=True)
class _Frame:
    packet: _Packet
    pose: np.ndarray
    tiles: dict


@dataclass(frozen=True)
class _Tile:
    keys: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    point_counts: np.ndarray
    keyframe_counts: np.ndarray


class ReversibleSemanticMap:
    """A bounded session of increasing, unique nonnegative int64 ns stamps.

    ``add`` returns ``updated``, ``duplicate_stamp``, ``nonmonotonic_stamp`` or
    ``capacity_reached``. Malformed payloads raise ValueError. No capacity
    eviction or point truncation occurs. ``update_poses`` accepts any ordered or
    unordered subset, or the complete graph history (at most max_keyframes),
    validates even unknown poses, and ignores unknown timestamps. Duplicate
    batch timestamps are errors. Exactly repeated poses are no-ops; every
    actual numeric change is honored, without a motion deadband.
    """

    def __init__(self, voxel=.2, classes=16, tile_voxels=20,
                 max_keyframes=1000, max_points=200000,
                 excluded_classes=DEFAULT_EXCLUDED_CLASSES,
                 minimum_confidence=0.0, max_total_points=4000000):
        classes = _integer(classes, 'classes', maximum=65536)
        tile_voxels = _integer(tile_voxels, 'tile_voxels')
        max_keyframes = _integer(max_keyframes, 'max_keyframes', maximum=1000)
        max_points = _integer(max_points, 'max_points')
        max_total_points = _integer(max_total_points, 'max_total_points')
        for name, value, low, high in (
                ('voxel', voxel, 0., np.inf),
                ('minimum_confidence', minimum_confidence, 0., 1.)):
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or not low <= value <= high
                    or (name == 'voxel' and value == 0)):
                raise ValueError(f'invalid {name}')
        if tile_voxels ** 3 * classes > _I64_MAX:
            raise ValueError('tile_voxels cubed * classes exceeds int64 encoding')
        try:
            excluded = tuple(sorted({_integer(k, 'excluded class', 0, classes-1)
                                     for k in excluded_classes}))
        except TypeError as error:
            raise ValueError('excluded_classes must be an iterable of class IDs') from error
        self._config = _Config(float(voxel), classes, tile_voxels, max_keyframes,
                               max_points, excluded, float(minimum_confidence),
                               max_total_points)
        self._excluded = np.zeros(classes, dtype=bool)
        self._excluded[list(excluded)] = True
        self._frames = {}
        self._tile_sources = {}
        self._tiles = {}
        self._stored_points = self._peak_stored_points = 0
        self._input_points = self._excluded_points = self._low_confidence_points = 0
        self._voxel_count = self._contribution_cells = self._tile_memberships = 0
        self._total_weight = 0.
        self._revision = 0
        self._last_dirty_tiles = 0
        self._last_dirty_snapshot_ms = 0.

    @property
    def voxel(self):
        return self._config.voxel

    @property
    def classes(self):
        return self._config.classes

    def _counts(self):
        return dict(keyframes=len(self._frames), stored_points=self._stored_points,
                    peak_stored_points=self._peak_stored_points,
                    max_total_points=self._config.max_total_points,
                    map_voxels=self._voxel_count, map_tiles=len(self._tiles),
                    evidence_points=self._stored_points, total_weight=self._total_weight,
                    revision=self._revision)

    def _report(self, start, status, **fields):
        result = dict(status=status, changed_keyframes=0, dirty_tiles=0,
                      reprojected_points=0, recomputed_keyframe_tiles=0,
                      recomputed_voxels=0, removed_voxels=0, added_voxels=0,
                      contribution_ms=0., tile_recompute_ms=0., dirty_snapshot_ms=0.)
        result.update(self._counts())
        result.update(fields)
        result['total_ms'] = (perf_counter()-start)*1000
        return result

    def _packet(self, points, labels, confidence):
        raw = _real_array(points, 'points')
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError('points must have shape (N, 3) in the pose body frame')
        n = len(raw)
        if n > self._config.max_points:
            return None, n
        labels = np.asarray(labels)
        confidence = _real_array(confidence, 'confidence')
        if labels.shape != (n,) or confidence.shape != (n,):
            raise ValueError('labels and confidence must have shape (N,)')
        if not n:
            labels = np.empty(0, dtype=np.uint16)
        if (labels.dtype.kind not in 'iu' or np.any(labels < 0)
                or np.any(labels >= self.classes)):
            raise ValueError('labels must be integer class IDs in [0, classes)')
        if (not np.isfinite(raw).all() or not np.isfinite(confidence).all()
                or np.any(confidence < 0) or np.any(confidence > 1)):
            raise ValueError('points must be finite; confidence must be finite in [0, 1]')
        confidence = np.array(confidence, dtype=np.float32, copy=True)
        excluded = self._excluded[labels]
        low = ((confidence < self._config.minimum_confidence) | (confidence <= 0)) & ~excluded
        keep = ~(excluded | low)
        packet = _Packet(_readonly(np.array(raw[keep], dtype=np.float64, copy=True)),
                         _readonly(np.array(labels[keep], dtype=np.uint16, copy=True)),
                         _readonly(confidence[keep]), n, int(excluded.sum()), int(low.sum()))
        return packet, n

    @staticmethod
    def _tile_index(coords):
        """Fast scalar unique for ordinary clouds; safe fallback for huge spans."""
        low, high = coords.min(axis=0), coords.max(axis=0)
        span = tuple(int(b)-int(a)+1 for a, b in zip(low, high))
        if span[0]*span[1]*span[2] > _I64_MAX:
            return np.unique(coords, axis=0, return_inverse=True)
        code = ((coords[:, 0]-low[0])*span[1] + coords[:, 1]-low[1])*span[2] + coords[:, 2]-low[2]
        unique, inverse = np.unique(code, return_inverse=True)
        tiles = np.column_stack((unique//(span[1]*span[2]) + low[0],
                                 unique//span[2] % span[1] + low[1],
                                 unique % span[2] + low[2]))
        return tiles, inverse

    def _build_contributions(self, packet, pose):
        if not len(packet.points):
            return {}
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            # Explicit 3D contraction avoids a threaded BLAS launch for an Nx3
            # matrix, so the semantic worker cannot monopolize BLAS threads.
            world = np.einsum('nj,ij->ni', packet.points, pose[:3, :3], optimize=False)
            grid = np.floor((world + pose[:3, 3])/self.voxel)
            if (not np.isfinite(grid).all() or np.any(grid < -(1 << 62))
                    or np.any(grid >= (1 << 62))
                    or not np.isfinite((grid+.5)*self.voxel).all()):
                raise ValueError('transformed geometry exceeds safe finite voxel coordinates')
        grid = grid.astype(np.int64)
        size = self._config.tile_voxels
        tile_coords = grid // size
        tiles, tile_rows = self._tile_index(tile_coords)
        local = grid-tile_coords*size
        codes = ((local[:, 0]*size + local[:, 1])*size + local[:, 2])*self.classes + packet.labels
        order = np.lexsort((codes, tile_rows))
        codes, rows = codes[order], tile_rows[order]
        starts = np.r_[0, np.flatnonzero((codes[1:] != codes[:-1]) | (rows[1:] != rows[:-1]))+1]
        weights = _readonly(np.add.reduceat(packet.confidence[order].astype(np.float64), starts))
        counts = _readonly(np.diff(np.r_[starts, len(order)]).astype(np.int64))
        codes, rows = _readonly(codes[starts]), rows[starts]
        boundaries = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1])+1, len(rows)]
        result = {}
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            key = tuple(int(v) for v in tiles[rows[begin]])
            cell_codes = codes[begin:end]
            result[key] = _Contribution(cell_codes, weights[begin:end], counts[begin:end],
                                        _readonly(np.unique(cell_codes//self.classes)))
        return result

    def _reduce_tile(self, key, sources):
        # Stable stamp order makes update order irrelevant, even at floating ties.
        pieces = [sources[stamp] for stamp in sorted(sources)]
        codes, inverse = np.unique(np.concatenate([p.codes for p in pieces]), return_inverse=True)
        weights = np.bincount(inverse, weights=np.concatenate([p.weights for p in pieces]))
        counts = np.bincount(inverse, weights=np.concatenate([p.counts for p in pieces]))
        voxels, row = np.unique(codes//self.classes, return_inverse=True)
        scores = np.zeros((len(voxels), self.classes), dtype=np.float64)
        scores[row, codes % self.classes] = weights
        point_counts = np.bincount(row, weights=counts).astype(np.int64)
        _, keyframe_counts = np.unique(np.concatenate([p.voxels for p in pieces]), return_counts=True)
        start = perf_counter()
        size = self._config.tile_voxels
        local = np.column_stack((voxels//(size*size), voxels//size % size, voxels % size))
        keys = local + np.asarray(key, dtype=np.int64)*size
        labels = scores.argmax(axis=1).astype(np.uint16)
        confidence = (scores[np.arange(len(labels)), labels]/scores.sum(axis=1)).astype(np.float32)
        tile = _Tile(*map(_readonly, (keys, scores, labels, confidence, point_counts,
                                     keyframe_counts.astype(np.int64))))
        return tile, (perf_counter()-start)*1000

    def _stage(self, replacements):
        """Construct every changed tile before publishing any replacement."""
        edits = {}
        for stamp, frame in replacements.items():
            previous = self._frames.get(stamp)
            if previous is not None:
                for key in previous.tiles:
                    edits.setdefault(key, {})[stamp] = None
            for key, contribution in frame.tiles.items():
                edits.setdefault(key, {})[stamp] = contribution
        start = perf_counter()
        sources, tiles = {}, {}
        snapshot_ms = 0.
        recomputed = removed = added = memberships = 0
        for key, changes in edits.items():
            candidates = self._tile_sources.get(key, {}).copy()
            for stamp, contribution in changes.items():
                if contribution is None:
                    candidates.pop(stamp, None)
                else:
                    candidates[stamp] = contribution
            sources[key] = candidates
            old_count = len(self._tiles[key].keys) if key in self._tiles else 0
            if candidates:
                tile, elapsed = self._reduce_tile(key, candidates)
                tiles[key] = tile
                snapshot_ms += elapsed
                new_count = len(tile.keys)
                memberships += len(candidates)
            else:
                tiles[key] = None
                new_count = 0
            recomputed += new_count
            # Counts refer to replaced tile snapshots, not set differences.
            removed += old_count
            added += new_count
        elapsed = (perf_counter()-start)*1000
        return sources, tiles, dict(dirty_tiles=len(edits),
                                   recomputed_keyframe_tiles=memberships,
                                   recomputed_voxels=recomputed, removed_voxels=removed,
                                   added_voxels=added, tile_recompute_ms=elapsed,
                                   dirty_snapshot_ms=snapshot_ms)

    def _commit(self, replacements, sources, tiles, metrics):
        for stamp, frame in replacements.items():
            previous = self._frames.get(stamp)
            if previous is not None:
                self._tile_memberships -= len(previous.tiles)
                self._contribution_cells -= sum(len(c.codes) for c in previous.tiles.values())
            self._frames[stamp] = frame
            self._tile_memberships += len(frame.tiles)
            self._contribution_cells += sum(len(c.codes) for c in frame.tiles.values())
        for key, tile in tiles.items():
            if tile is None:
                self._tiles.pop(key, None)
                self._tile_sources.pop(key, None)
            else:
                self._tiles[key] = tile
                self._tile_sources[key] = sources[key]
        self._voxel_count += metrics['added_voxels']-metrics['removed_voxels']
        self._last_dirty_tiles = metrics['dirty_tiles']
        self._last_dirty_snapshot_ms = metrics['dirty_snapshot_ms']
        self._revision += 1

    def add(self, stamp_ns, points, labels, confidence, pose):
        """Add one local packet; preserve all eligible observations without sampling."""
        start = perf_counter()
        stamp = _integer(stamp_ns, 'stamp_ns', minimum=0)
        if stamp in self._frames:
            return self._report(start, 'duplicate_stamp', stamp_ns=stamp)
        if self._frames and stamp < next(reversed(self._frames)):
            return self._report(start, 'nonmonotonic_stamp', stamp_ns=stamp)
        if len(self._frames) >= self._config.max_keyframes:
            return self._report(start, 'capacity_reached', reason='max_keyframes', stamp_ns=stamp)
        packet, input_points = self._packet(points, labels, confidence)
        if packet is None:
            return self._report(start, 'capacity_reached', reason='max_points',
                                stamp_ns=stamp, input_points=input_points)
        poses = _poses(np.asarray(pose)[None], 1)
        if self._stored_points+len(packet.points) > self._config.max_total_points:
            return self._report(start, 'capacity_reached', reason='max_total_points',
                                stamp_ns=stamp, input_points=input_points,
                                requested_stored_points=len(packet.points))
        contribution_start = perf_counter()
        frame = _Frame(packet, poses[0], self._build_contributions(packet, poses[0]))
        contribution_ms = (perf_counter()-contribution_start)*1000
        sources, tiles, metrics = self._stage({stamp: frame})
        self._commit({stamp: frame}, sources, tiles, metrics)
        self._stored_points += len(packet.points)
        self._peak_stored_points = max(self._peak_stored_points, self._stored_points)
        self._input_points += packet.input_points
        self._excluded_points += packet.excluded_points
        self._low_confidence_points += packet.low_confidence_points
        self._total_weight += float(packet.confidence.sum(dtype=np.float64))
        return self._report(start, 'updated', stamp_ns=stamp, input_points=input_points,
                            accepted_points=len(packet.points), changed_keyframes=1,
                            reprojected_points=len(packet.points),
                            contribution_ms=contribution_ms, **metrics)

    def update_poses(self, times_ns, poses):
        """Atomically revise known stamps in a validated, bounded graph pose batch."""
        start = perf_counter()
        raw_times = np.asarray(times_ns)
        if raw_times.ndim != 1 or len(raw_times) > self._config.max_keyframes:
            raise ValueError('times_ns must be a 1D batch of at most max_keyframes stamps')
        times = [_integer(t, 'times_ns stamp', minimum=0) for t in times_ns]
        if len(set(times)) != len(times):
            raise ValueError('duplicate timestamps in pose update')
        poses = _poses(poses, len(times))
        replacements = {}
        unknown = 0
        points = 0
        contribution_start = perf_counter()
        for stamp, pose in zip(times, poses):
            previous = self._frames.get(stamp)
            if previous is None:
                unknown += 1
            elif not np.array_equal(previous.pose, pose):
                packet = previous.packet
                replacements[stamp] = _Frame(packet, pose, self._build_contributions(packet, pose))
                points += len(packet.points)
        contribution_ms = (perf_counter()-contribution_start)*1000
        if not replacements:
            return self._report(start, 'unchanged', unknown_keyframes=unknown,
                                contribution_ms=contribution_ms)
        sources, tiles, metrics = self._stage(replacements)
        self._commit(replacements, sources, tiles, metrics)
        return self._report(start, 'updated', changed_keyframes=len(replacements),
                            unknown_keyframes=unknown, reprojected_points=points,
                            contribution_ms=contribution_ms, **metrics)

    def snapshot(self, max_points=None, *, include_scores=True):
        """Return detached arrays sorted by (x,y,z) voxel key.

        ``max_points`` samples evenly from that order for display only; it never
        limits stored evidence or full-map counts. Sorting/gathering a complete
        display snapshot is O(map size), separate from the tile-local update.
        ``include_scores=False`` avoids copying the full dense vote matrix.
        """
        start = perf_counter()
        if max_points is not None:
            max_points = _integer(max_points, 'max_points', minimum=0)
        tiles = list(self._tiles.values())
        keys = np.concatenate([t.keys for t in tiles]) if tiles else np.empty((0, 3), np.int64)
        sort_start = perf_counter()
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        if max_points is not None and max_points < len(order):
            sample = np.arange(max_points, dtype=np.int64)*len(order)//max_points if max_points else []
            order = order[sample]
        sort_ms = (perf_counter()-sort_start)*1000
        keys = keys[order]
        result = dict(keys=keys, points=(keys.astype(np.float64)+.5)*self.voxel)
        fields = dict(labels=(np.uint16, (0,)), confidence=(np.float32, (0,)),
                      point_counts=(np.int64, (0,)), keyframe_counts=(np.int64, (0,)))
        if include_scores:
            fields['scores'] = (np.float64, (0, self.classes))
        for name, (dtype, shape) in fields.items():
            result[name] = (np.concatenate([getattr(t, name) for t in tiles])[order]
                            if tiles else np.empty(shape, dtype=dtype))
        result.update(self._counts())
        result.update(display_points=len(keys), sort_ms=sort_ms,
                      dirty_tiles=self._last_dirty_tiles,
                      dirty_snapshot_ms=self._last_dirty_snapshot_ms,
                      snapshot_ms=(perf_counter()-start)*1000)
        return result

    def summary(self):
        """O(1) evidence/work counts; display caps never change map_voxels."""
        result = self._counts()
        result.update(input_points=self._input_points,
                      filtered_points=self._input_points-self._stored_points,
                      excluded_points=self._excluded_points,
                      low_confidence_points=self._low_confidence_points,
                      contribution_cells=self._contribution_cells,
                      keyframe_tiles=self._tile_memberships,
                      max_keyframes=self._config.max_keyframes,
                      max_points=self._config.max_points, voxel=self.voxel,
                      classes=self.classes, tile_voxels=self._config.tile_voxels,
                      excluded_classes=self._config.excluded_classes,
                      minimum_confidence=self._config.minimum_confidence,
                      first_stamp_ns=next(iter(self._frames), None),
                      last_stamp_ns=next(reversed(self._frames), None),
                      dirty_tiles=self._last_dirty_tiles,
                      dirty_snapshot_ms=self._last_dirty_snapshot_ms)
        return result

    def verify_rebuild(self, *, rtol=1e-12, atol=1e-12):
        """OFFLINE diagnostic: rebuild the entire history at its latest poses.

        Uses the identical constructor configuration, including both point caps.
        It does not change this map and is intentionally not a runtime update.
        This checks reversibility; independent algorithm tests must additionally
        compare to a reference reducer to catch shared implementation mistakes.
        """
        start = perf_counter()
        for tolerance in (rtol, atol):
            if (isinstance(tolerance, (bool, np.bool_))
                    or not isinstance(tolerance, (int, float, np.integer, np.floating))
                    or not np.isfinite(tolerance) or tolerance < 0):
                raise ValueError('parity tolerances must be finite and nonnegative')
        rebuilt = ReversibleSemanticMap(**vars(self._config))
        for stamp, frame in self._frames.items():
            p = frame.packet
            report = rebuilt.add(stamp, p.points, p.labels, p.confidence, frame.pose)
            if report['status'] != 'updated':
                raise RuntimeError(f'configured rebuild rejected stamp {stamp}: {report}')
        rebuild_ms = (perf_counter()-start)*1000
        actual, expected = self.snapshot(), rebuilt.snapshot()
        compare_start = perf_counter()
        keys_equal = np.array_equal(actual['keys'], expected['keys'])
        checks = dict(keys_equal=keys_equal)
        for name in ('labels', 'point_counts', 'keyframe_counts'):
            checks[name+'_equal'] = keys_equal and np.array_equal(actual[name], expected[name])
        for name in ('scores', 'confidence'):
            checks[name+'_equal'] = keys_equal and np.allclose(actual[name], expected[name], rtol=rtol, atol=atol)
        errors = [name for name, passed in checks.items() if not passed]
        max_score_error = (float(np.max(np.abs(actual['scores']-expected['scores']), initial=0))
                           if keys_equal else None)
        max_confidence_error = (float(np.max(np.abs(actual['confidence']-expected['confidence']), initial=0))
                                if keys_equal else None)
        return dict(ok=not errors, errors=errors, **checks,
                    keyframes=len(self._frames), stored_points=self._stored_points,
                    map_voxels=self._voxel_count, rebuilt_voxels=rebuilt._voxel_count,
                    compared_voxels=self._voxel_count if keys_equal else 0,
                    max_score_error=max_score_error, max_confidence_error=max_confidence_error,
                    rebuild_ms=rebuild_ms,
                    snapshot_ms=actual['snapshot_ms']+expected['snapshot_ms'],
                    comparison_ms=(perf_counter()-compare_start)*1000,
                    total_ms=(perf_counter()-start)*1000)
