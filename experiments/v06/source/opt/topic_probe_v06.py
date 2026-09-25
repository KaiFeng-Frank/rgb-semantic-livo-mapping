#!/usr/bin/env python3
"""
opt/topic_probe_v06.py -- subscriber-side counter for /semantic_scan and /semantic_map.

Records (header stamp, wall arrival, points, bytes) per received message and nothing
else -- no `ros2 topic hz` (its own CPU would pollute the contention measurement).  QoS
matches the publishers exactly: /semantic_scan BEST_EFFORT/KEEP_LAST, /semantic_map
RELIABLE/TRANSIENT_LOCAL/KEEP_LAST.  Written as an npz when the run is over (SIGINT or
--idle seconds without any message once one has arrived).
"""
import argparse, os, signal, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.clock import Clock, ClockType
from sensor_msgs.msg import PointCloud2


class Probe(Node):
    def __init__(self, a):
        super().__init__("topic_probe_v06")
        self.a = a
        self.scan = []
        self.map = []
        self.last = None
        self.done = False
        self.create_subscription(
            PointCloud2, "/semantic_scan", self.cb_scan,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=a.scan_depth))
        self.create_subscription(
            PointCloud2, "/semantic_map", self.cb_map,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=a.map_depth,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(1.0, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def cb_scan(self, m):
        self.scan.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, time.time(),
                          m.width * m.height, len(m.data)))
        self.last = time.time()

    def cb_map(self, m):
        self.map.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, time.time(),
                         m.width * m.height, len(m.data)))
        self.last = time.time()

    def tick(self):
        if self.last is not None and time.time() - self.last > self.a.idle and not self.done:
            self.finish()
            raise SystemExit

    def finish(self):
        if self.done:
            return
        self.done = True
        np.savez_compressed(self.a.out,
                            scan=np.array(self.scan, dtype=np.float64).reshape(-1, 4),
                            map=np.array(self.map, dtype=np.float64).reshape(-1, 4))
        print("probe: %d scan msgs, %d map msgs -> %s" % (len(self.scan), len(self.map), self.a.out),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--idle", type=float, default=20.0)
    ap.add_argument("--scan-depth", type=int, default=10)
    ap.add_argument("--map-depth", type=int, default=5)
    a = ap.parse_args()
    rclpy.init()
    n = Probe(a)
    signal.signal(signal.SIGTERM, lambda *_: (n.finish(), os._exit(0)))
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        n.finish()
        try:
            n.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
