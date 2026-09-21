#!/usr/bin/env python3
"""publish_npz.py -- republish a saved map .npz on /semantic_map for RViz.

Used only for the S6 visual: it lets the SAME map be shown with and without the
v0.3 publish filter, so the two screenshots differ by the filter alone and not by
PTv3's run-to-run non-determinism.
"""
import argparse, os, sys, time
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from semantic_map_node import make_cloud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--frame", default="map")
    ap.add_argument("--seconds", type=float, default=25.0)
    a = ap.parse_args()
    z = np.load(a.npz)
    rec = S.build_record(z["xyz"], z["rgb"], z["cls"], z["conf"], z["has_rgb"])
    rclpy.init()
    n = Node("npz_pub")
    q = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                   depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = n.create_publisher(PointCloud2, "/semantic_map", q)
    msg = make_cloud(rec, a.frame, n.get_clock().now().to_msg())
    t0 = time.time()
    print("publishing %d points from %s" % (len(rec), os.path.basename(a.npz)), flush=True)
    while time.time() - t0 < a.seconds:
        pub.publish(msg)
        rclpy.spin_once(n, timeout_sec=0.5)
        time.sleep(1.0)
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
