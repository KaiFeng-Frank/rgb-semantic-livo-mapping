"""Owned, persistent GPU subprocess; ROS does not import torch."""
import argparse
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys
import time


class Client:
    def __init__(self, python, weights, log, imgsz=640):
        from multiprocessing import Pipe
        self.conn, child = Pipe()
        self.log = Path(log).open('w')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   YOLO_AUTOINSTALL='false')
        env['PYTHONPATH'] = env.get('VISUAL_PYTHONPATH', '')
        self.process = subprocess.Popen([str(python), '-m', 'tools.visual_instances.worker', '--fd', str(child.fileno()),
            '--weights', str(weights), '--imgsz', str(imgsz)], pass_fds=[child.fileno()],
            cwd=Path(__file__).resolve().parents[2], stdout=self.log, stderr=subprocess.STDOUT,
            env=env)
        child.close()
        try:
            if not self.conn.poll(90):
                raise TimeoutError('Visual worker startup deadline')
            self.meta = self.conn.recv()
            if 'error' in self.meta:
                raise RuntimeError(self.meta['error'])
        except BaseException:
            self.close()
            raise

    def step(self, image, stamp_ns):
        self.conn.send((image, stamp_ns))
        if not self.conn.poll(3):
            raise TimeoutError('Visual inference deadline')
        result = self.conn.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

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
    args = p.parse_args()
    os.nice(5)
    conn = Connection(args.fd)
    try:
        from .detector import Detector
        model = Detector(args.weights, imgsz=args.imgsz)
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
