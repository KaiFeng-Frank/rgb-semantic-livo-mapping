"""Bounded online image consumer, GPU worker, and instance-packet publisher."""
import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from .worker import Client


def make_node(args):
    import cv2
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import String
    cv2.setNumThreads(1)

    class VisualNode(Node):
        def __init__(self):
            super().__init__('visual_instance_association')
            args.out.mkdir(parents=True, exist_ok=False)
            self.log = (args.out/'events.jsonl').open('w')
            self.lock, self.stop, self.wake = threading.Lock(), threading.Event(), threading.Event()
            self.pending, self.last_stamp, self.client = None, -1, None
            self.counts = dict(received=0, processed=0, superseded=0, invalid=0, stale=0,
                               ready=False, failed=False, error=None)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub = self.create_publisher(String, '/backend/visual_instances', qos)
            self.status_pub = self.create_publisher(String, '/backend/visual_status', 1)
            self.create_subscription(CompressedImage if args.compressed else Image, args.image_topic, self.on_image, qos)
            self.create_timer(.5, self.status)
            self.thread = threading.Thread(target=self.work, daemon=True, name='visual-instances')
            self.thread.start()

        def on_image(self, message):
            stamp = int(message.header.stamp.sec*10**9+message.header.stamp.nanosec)
            if stamp <= self.last_stamp or len(message.data) > 16_000_000:
                self.counts['invalid'] += 1
                return
            self.last_stamp = stamp
            with self.lock:
                self.counts['received'] += 1
                if self.pending is not None:
                    self.counts['superseded'] += 1
                self.pending = (stamp, message, time.monotonic())
            self.wake.set()

        def status(self):
            with self.lock:
                row = dict(self.counts, pending=int(self.pending is not None))
            self.status_pub.publish(String(data=json.dumps(row)))
            return row

        def work(self):
            try:
                self.client = Client(args.worker_python, args.weights, args.out/'worker.log', args.imgsz, cancel_event=self.stop,
                                     feature_embeddings=getattr(args,'feature_embeddings',False))
                (args.out/'model.json').write_text(json.dumps(self.client.meta, indent=2)+'\n')
                self.counts['ready'] = True
                while not self.stop.is_set():
                    self.wake.wait(.1); self.wake.clear()
                    with self.lock:
                        packet, self.pending = self.pending, None
                    if packet is None:
                        continue
                    stamp, message, arrival = packet
                    if time.monotonic()-arrival > .3:
                        self.counts['stale'] += 1
                        continue
                    if args.compressed:
                        pixels = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
                    else:
                        if message.encoding not in ('bgr8', 'rgb8'):
                            raise ValueError('Only bgr8/rgb8 raw images supported')
                        pixels = np.ndarray((message.height, message.width, 3), np.uint8,
                            buffer=message.data, strides=(message.step, 3, 1))
                        if message.encoding == 'rgb8':
                            pixels = pixels[:, :, ::-1]
                    if pixels is None or pixels.shape[:2] != (args.height, args.width):
                        self.counts['invalid'] += 1
                        continue
                    result = self.client.step(np.ascontiguousarray(pixels), stamp)
                    self.pub.publish(String(data=result['packet']))
                    self.counts['processed'] += 1
                    self.log.write(json.dumps(dict(stamp_ns=stamp, arrival=arrival,
                        published=time.monotonic(), receipt_to_publish_ms=(time.monotonic()-arrival)*1000,
                        **result['info']))+'\n'); self.log.flush()
            except InterruptedError as exc:
                if not self.stop.is_set():
                    self.counts.update(failed=True, error=f'{type(exc).__name__}: {exc}')
            except Exception as exc:
                self.counts.update(failed=True, error=f'{type(exc).__name__}: {exc}')
            finally:
                self.counts['ready'] = False
                if self.client:
                    self.client.close()

        def close(self):
            self.stop.set(); self.wake.set(); self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise TimeoutError('Visual worker failed to stop')
            (args.out/'final.json').write_text(json.dumps(self.status(), indent=2)+'\n')
            self.log.close()

    return VisualNode()


def main():
    import rclpy
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-python', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--image-topic', default='/left_camera/image/compressed')
    p.add_argument('--compressed', action='store_true')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--feature-embeddings',action='store_true')
    p.add_argument('--frontend', type=Path, help='Read image topic/transport from existing frontend config')
    p.add_argument('--camera', type=Path, help='Read raw image dimensions from existing camera config')
    args = p.parse_args()
    os.nice(10)
    # Explicit deployment resource assignment, inherited by the GPU worker.
    # No global scheduler changes and no guessed physical CPU topology.
    if os.environ.get('VISUAL_CPU_AFFINITY'):
        os.sched_setaffinity(0, {int(v) for v in os.environ['VISUAL_CPU_AFFINITY'].split(',')})
    if args.frontend:
        import yaml
        common = yaml.safe_load(args.frontend.read_text())['/**']['ros__parameters']['common']
        args.image_topic, args.compressed = common['img_topic'], common.get('img_compressed', False)
    if args.camera:
        import yaml
        camera = yaml.safe_load(args.camera.read_text())['/**']['ros__parameters']
        args.width, args.height = camera['cam_width'], camera['cam_height']
    rclpy.init()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    node = make_node(args)
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=.02)
    except KeyboardInterrupt:
        pass
    finally:
        node.close(); node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
