"""ROS-free bounded IPC. Native optimization runs outside the frontend/ROS process."""
import argparse
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_settings(path=None):
    here = Path(__file__).parent
    cfg = json.loads((here/'protocol.json').read_text())
    cfg.update(json.loads((here/'online.json').read_text()))
    if path:
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


class ExactPairs:
    """Never match by arrival order, nearest timestamp, or future interpolation."""
    def __init__(self, capacity):
        self.capacity, self.poses, self.clouds, self.evicted = capacity, {}, {}, 0

    def add(self, kind, stamp, value):
        mine, other = (self.poses, self.clouds) if kind == 'pose' else (self.clouds, self.poses)
        if stamp in other:
            match = other.pop(stamp)
            return (value, match) if kind == 'pose' else (match, value)
        mine[stamp] = value
        while len(mine) > self.capacity:
            del mine[min(mine)]
            self.evicted += 1
        return None


class BackendProcess:
    def __init__(self, python, cfg, log=None):
        self.cfg = cfg
        self.pending = queue.Queue(maxsize=cfg['pending_keyframes'])
        self.lock = threading.Lock()
        self.io_lock = threading.Lock()
        self.ready, self.stop = threading.Event(), threading.Event()
        self.latest, self.error = None, None
        self.stats = dict(submitted=0, completed=0, queue_full=0, stale=0, late_results=0, not_ready=0)
        self.io_deadline = time.monotonic()+cfg['worker_timeout_s']
        parent, child = socket.socketpair()
        fd = child.fileno()
        command = [str(python), '-m', 'tools.loop_closure.online_transport', '--worker-fd', str(fd)]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, env=env, pass_fds=(fd,))
        except Exception:
            parent.close()
            raise
        finally:
            child.close()
        self.connection = Connection(parent.detach())
        self.connection.send(dict(cfg=cfg, log=str(log) if log else None))
        self.thread = threading.Thread(target=self._exchange, name='backend-ipc', daemon=True)
        self.thread.start()
        self.watchdog = threading.Thread(target=self._watch, name='backend-watchdog', daemon=True)
        self.watchdog.start()

    def _watch(self):
        # poll()+recv() alone has no complete-message deadline: a partial
        # payload can block recv indefinitely. Killing our owned worker closes
        # the peer socket and releases either send or recv.
        while not self.stop.wait(.05):
            with self.io_lock:
                deadline = self.io_deadline
                if deadline is not None and time.monotonic() > deadline:
                    self.error = 'TimeoutError: complete backend exchange deadline exceeded'
                    self.ready.clear()
                    if self.process.poll() is None:
                        self.process.kill()
                    return

    def _deadline(self, value):
        with self.io_lock:
            self.io_deadline = value

    def _receive(self):
        deadline = time.monotonic()+self.cfg['worker_timeout_s']
        while not self.stop.is_set():
            if self.connection.poll(.05):
                return self.connection.recv()
            if self.process.poll() is not None:
                raise RuntimeError(f'backend exited with {self.process.returncode}')
            if time.monotonic() > deadline:
                raise TimeoutError('backend worker response deadline exceeded')
        raise InterruptedError('backend stopped')

    def _exchange(self):
        try:
            reply = self._receive()
            if reply.get('status') != 'ready':
                raise RuntimeError(str(reply))
            self._deadline(None)
            self.ready.set()
            while not self.stop.is_set():
                try:
                    item = self.pending.get(timeout=.05)
                except queue.Empty:
                    if self.process.poll() is not None:
                        raise RuntimeError(f'backend exited with {self.process.returncode}')
                    continue
                if time.monotonic()-item['enqueued'] > self.cfg['max_queue_age_s']:
                    self.stats['stale'] += 1
                    continue
                self._deadline(time.monotonic()+self.cfg['worker_timeout_s'])
                self.connection.send(item)
                reply = self._receive()
                self._deadline(None)
                if reply.get('status') == 'error':
                    raise RuntimeError(reply['error'])
                self.stats['completed'] += 1
                reply['received_monotonic'] = time.monotonic()
                reply['queue_and_processing_ms'] = (time.monotonic()-item['enqueued'])*1000
                reply['late'] = reply['queue_and_processing_ms'] > self.cfg['max_queue_age_s']*1000
                self.stats['late_results'] += int(reply['late'])
                with self.lock:
                    if reply['status'] == 'updated':
                        # Keep the last complete map if this update carries only
                        # poses. Map generation/version is explicit for consumers.
                        if reply['map_points'] is None and self.latest:
                            reply['map_points'] = self.latest.get('map_points')
                            reply['map_stamp_ns'] = self.latest.get('map_stamp_ns')
                            reply['map_version'] = self.latest.get('map_version')
                        else:
                            reply['map_stamp_ns'] = reply['stamp_ns']
                            reply['map_version'] = reply['version']
                        self.latest = reply
                    elif self.latest:
                        self.latest = dict(self.latest, status=reply['status'])
        except InterruptedError:
            pass
        except Exception as exc:
            self.error = self.error or f'{type(exc).__name__}: {exc}'
        finally:
            self.ready.clear()
            if self.error and self.process.poll() is None:
                self.process.terminate()

    def submit(self, stamp_ns, pose, points):
        if not self.ready.is_set() or self.error:
            self.stats['not_ready'] += 1
            return False
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) > self.cfg['max_points_per_keyframe']:
            raise ValueError('keyframe cloud exceeds IPC contract')
        item = dict(stamp_ns=int(stamp_ns), pose=np.asarray(pose).copy(),
                    points=points.copy(), enqueued=time.monotonic())
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            # Keep current work plus the newest bounded pending observations.
            # Missing intervals are already reflected in graph edge variance.
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(item)
            self.stats['queue_full'] += 1
        self.stats['submitted'] += 1
        return True

    def snapshot(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        # This process belongs to this client; never touch other ROS/experiment jobs.
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.thread.join(timeout=2)
        self.watchdog.join(timeout=1)
        self.connection.close()


def worker(fd):
    os.nice(10)
    connection = Connection(fd)
    handle = None
    try:
        config = connection.recv()
        from .online import OnlineBackend, configuration
        # Validate defaults and the complete caller config through the core.
        cfg = configuration()
        cfg.update(config['cfg'])
        backend = OnlineBackend(cfg)
        if config['log']:
            path = Path(config['log'])
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('x')
            handle.write(json.dumps(dict(event='startup', config=cfg))+'\n')
            handle.flush()
        connection.send(dict(status='ready'))
        while True:
            request = connection.recv()
            result = backend.ingest(request['stamp_ns'], request['pose'], request['points'])
            if handle:
                row = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
                if 'correction' in result:
                    row['correction'] = result['correction'].tolist()
                row['event'] = 'keyframe'
                handle.write(json.dumps(row, allow_nan=False)+'\n')
                handle.flush()
            connection.send(result)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(dict(status='error', error=f'{type(exc).__name__}: {exc}'))
        except (OSError, EOFError):
            pass
    finally:
        if handle:
            handle.close()
        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-fd', type=int, required=True)
    worker(parser.parse_args().worker_fd)
