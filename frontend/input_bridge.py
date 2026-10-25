"""Preserve sensor geometry/time while adding causal semantic observations.

The deadline buffer is independent of ROS/CUDA so timing contracts can be tested.
The ROS adapter performs raw-scan PTv3 inference in a bounded background queue.
An expired scan is emitted once with unknown labels; late results are discarded.
"""
from collections import deque
from dataclasses import dataclass
import math


@dataclass
class Pending:
    key: int
    payload: object
    arrived: float
    deadline: float
    result: object = None
    completed: float = None


class DeadlineBuffer:
    def __init__(self, wait_ms=80., capacity=4, fixed_delay=False):
        if not math.isfinite(wait_ms) or wait_ms < 0 or capacity < 1:
            raise ValueError('invalid semantic deadline/capacity')
        self.wait = wait_ms / 1000.
        self.capacity = capacity
        self.fixed_delay = fixed_delay
        self.pending = deque()
        self.last_key = -1
        self.rejected_late = 0

    def push(self, key, payload, now):
        if key <= self.last_key:
            raise ValueError('duplicate/backwards scan timestamp; restart on clock reset')
        if len(self.pending) >= self.capacity:
            raise BufferError('drain oldest geometry scan before accepting another')
        self.last_key = key
        self.pending.append(Pending(key, payload, now, now+self.wait))

    def complete(self, key, result, now):
        for p in self.pending:
            if p.key == key and p.result is None and now <= p.deadline:
                p.result, p.completed = result, now
                return True
        self.rejected_late += 1
        return False

    def drain(self, now, force_oldest=False):
        out = []
        while self.pending:
            p = self.pending[0]
            if self.fixed_delay and now < p.deadline and not force_oldest:
                break
            if p.result is None and now < p.deadline and not force_oldest:
                break
            self.pending.popleft()
            out.append(p)
            if force_oldest:
                break
        return out


def add_fields(msg, labels=None, confidence=None):
    """Append fields without changing coordinates, intensity, ring, time or header."""
    import copy
    import numpy as np
    from sensor_msgs.msg import PointField
    if msg.is_bigendian:
        raise ValueError('little-endian point clouds required')
    if any(f.name.startswith('semantic_') for f in msg.fields):
        raise ValueError('input is already annotated')
    n = msg.width * msg.height
    if labels is None:
        labels = np.full(n, 65535, dtype='<u2')
        confidence = np.zeros(n, dtype='<f4')
    labels, confidence = np.asarray(labels), np.asarray(confidence)
    if labels.shape != (n,) or confidence.shape != (n,):
        raise ValueError('prediction count does not match original scan')
    valid = (labels >= 0) & (labels < 16) & np.isfinite(confidence) & (confidence >= 0) & (confidence <= 1)
    step = msg.point_step + 8
    data = np.zeros((n, step), dtype=np.uint8)
    original = np.ndarray((msg.height, msg.width, msg.point_step), dtype=np.uint8,
                          buffer=msg.data, strides=(msg.row_step, msg.point_step, 1))
    data[:, :msg.point_step] = original.reshape(n, msg.point_step)
    data[:, msg.point_step:msg.point_step+2] = np.where(valid, labels, 65535).astype('<u2').view(np.uint8).reshape(n, 2)
    data[:, msg.point_step+4:msg.point_step+8] = np.where(valid, confidence, 0).astype('<f4').view(np.uint8).reshape(n, 4)
    out = copy.deepcopy(msg)
    out.fields.extend([PointField(name='semantic_label', offset=msg.point_step, datatype=PointField.UINT16, count=1),
                       PointField(name='semantic_confidence', offset=msg.point_step+4, datatype=PointField.FLOAT32, count=1)])
    out.point_step, out.row_step = step, step*msg.width
    out.data = data.tobytes()
    return out


def main():
    import argparse
    import json
    from pathlib import Path
    import queue
    import sys
    import threading
    import time
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py.point_cloud2 import read_points
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='/velodyne_points')
    ap.add_argument('--output', default='/semantic_frontend/points')
    ap.add_argument('--mode', choices=['online', 'geometry'], default='online')
    ap.add_argument('--wait-ms', type=float, default=80.)
    ap.add_argument('--capacity', type=int, default=4)
    ap.add_argument('--fixed-delay', action='store_true', help='emit at the same deadline in semantic and geometry timing controls')
    ap.add_argument('--log', type=Path, required=True)
    args, ros_args = ap.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    worker = None
    if args.mode == 'online':
        from ptv3_client import PTv3Client
        worker = PTv3Client(half=1, shuffle=0)
        if worker.num_classes != 16:
            worker.close()
            raise RuntimeError('frontend taxonomy requires native nuScenes 16 classes')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open('w', buffering=1)
    rclpy.init(args=ros_args)

    class Bridge(Node):
        def __init__(self):
            super().__init__('semantic_frontend_input')
            self.buffer = DeadlineBuffer(args.wait_ms, args.capacity, args.fixed_delay)
            self.jobs = queue.Queue(maxsize=1)
            self.results = queue.Queue(maxsize=args.capacity)
            self.stopping = threading.Event()
            self.pub = self.create_publisher(PointCloud2, args.output, 10)
            self.sub = self.create_subscription(PointCloud2, args.input, self.receive, qos_profile_sensor_data)
            self.timer = self.create_timer(.002, self.tick)
            self.thread = threading.Thread(target=self.infer, daemon=True)
            self.thread.start()

        def publish(self, pending):
            result = pending.result
            self.pub.publish(add_fields(pending.payload, *(result if result is not None else (None, None))))
            log.write(json.dumps(dict(stamp_ns=pending.key, semantic=result is not None,
                                      arrival=pending.arrived, deadline=pending.deadline,
                                      inference_complete=pending.completed, emitted=time.monotonic(),
                                      mode=args.mode, late_discarded=self.buffer.rejected_late))+'\n')

        def receive(self, msg):
            self.tick()
            if len(self.buffer.pending) >= args.capacity:
                for p in self.buffer.drain(time.monotonic(), force_oldest=True): self.publish(p)
            key = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
            now = time.monotonic()
            try:
                self.buffer.push(key, msg, now)
            except ValueError as exc:
                self.get_logger().error(str(exc))
                return
            if worker is not None:
                try: self.jobs.put_nowait((key, msg, now+self.buffer.wait))
                except queue.Full: pass  # bounded compute: still emit geometry once

        def infer(self):
            while not self.stopping.is_set():
                try: key, msg, deadline = self.jobs.get(timeout=.1)
                except queue.Empty: continue
                if time.monotonic() > deadline: continue
                try:
                    arr = read_points(msg, field_names=('x','y','z','intensity'), skip_nans=False).reshape(-1)
                    pts = np.column_stack([arr[k] for k in ('x','y','z','intensity')]).astype(np.float32)
                    valid = np.isfinite(pts).all(axis=1)
                    labels = np.full(len(pts), 65535, dtype=np.uint16)
                    conf = np.zeros(len(pts), dtype=np.float32)
                    if valid.sum() >= 2:
                        labels[valid], conf[valid] = worker.segment(pts[valid])
                    self.results.put_nowait((key, (labels,conf), time.monotonic()))
                except Exception as exc:
                    self.get_logger().error('semantic inference failed: '+str(exc))

        def tick(self):
            while True:
                try: key, result, when = self.results.get_nowait()
                except queue.Empty: break
                self.buffer.complete(key, result, when)
            for p in self.buffer.drain(time.monotonic()): self.publish(p)

    node = Bridge()
    try:
        rclpy.spin(node)
    finally:
        node.stopping.set()
        node.thread.join(timeout=2.)
        if worker is not None: worker.close()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        log.close()


if __name__ == '__main__':
    main()
