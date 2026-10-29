"""Owned, persistent GPU subprocess; ROS does not import torch."""
import argparse
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys
import time


class Client:
    def __init__(self, python, weights, log, imgsz=640, cancel_event=None, feature_embeddings=False,fullrate=False,visual_history=False):
        from multiprocessing import Pipe
        self.conn, child = Pipe()
        self.cancel_event = cancel_event
        self.log = Path(log).open('w')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   YOLO_AUTOINSTALL='false')
        env['PYTHONPATH'] = env.get('VISUAL_PYTHONPATH', '')
        self.process = subprocess.Popen([str(python), '-m', 'tools.visual_instances.worker', '--fd', str(child.fileno()),
            '--weights', str(weights), '--imgsz', str(imgsz),
            *(['--feature-embeddings'] if feature_embeddings else []),
            *(['--fullrate'] if fullrate else []),*(['--visual-history'] if visual_history else [])], pass_fds=[child.fileno()],
            cwd=Path(__file__).resolve().parents[2], stdout=self.log, stderr=subprocess.STDOUT,
            env=env)
        child.close()
        try:
            self.wait(90, 'startup')
            self.meta = self.conn.recv()
            if 'error' in self.meta:
                raise RuntimeError(self.meta['error'])
        except BaseException:
            self.close()
            raise

    def step(self, image, stamp_ns):
        self.conn.send((image, stamp_ns))
        self.wait(3, 'inference')
        result = self.conn.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def wait(self, seconds, phase):
        deadline = time.monotonic()+seconds
        while not self.conn.poll(.05):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise InterruptedError('Visual worker cancelled')
            if self.process.poll() is not None:
                raise RuntimeError('Visual worker exited')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Visual {phase} deadline')

    def close(self):
        self.conn.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=3)
        self.log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fd', type=int, required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--feature-embeddings',action='store_true')
    p.add_argument('--fullrate',action='store_true')
    p.add_argument('--visual-history',action='store_true')
    args = p.parse_args()
    os.nice(5)
    conn = Connection(args.fd)
    try:
        if args.visual_history and not args.fullrate:raise ValueError('Visual history requires the explicit full-rate mode')
        if args.fullrate:
            from .fullrate import FullRateDetector
            model=FullRateDetector(args.weights,imgsz=args.imgsz,feature_embeddings=args.feature_embeddings,visual_history=args.visual_history)
        else:
            from .detector import Detector
            model = Detector(args.weights, imgsz=args.imgsz,feature_embeddings=args.feature_embeddings)
        conn.send(model.meta)
        while True:
            image, stamp = conn.recv()
            frame, info = model.step(image, stamp)
            conn.send(dict(packet=frame.encode(), info=info))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        conn.send(dict(error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
