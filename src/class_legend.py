#!/usr/bin/env python3
"""Publish a 16-block cloud (class = 0..15) so RViz's Intensity/class colouring
can be read off exactly instead of guessed."""
import sys, time, numpy as np, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from semantic_map_node import make_cloud
rclpy.init(); n = rclpy.create_node("class_legend")
q = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
               depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
from sensor_msgs.msg import PointCloud2
p = n.create_publisher(PointCloud2, "/class_legend", q)
X, C = [], []
for i in range(16):
    g = np.mgrid[0:30, 0:30].reshape(2, -1).T * 0.1
    X.append(np.column_stack([g[:, 0] + i * 4.0, g[:, 1], np.zeros(len(g))]))
    C.append(np.full(len(g), i, np.uint16))
X = np.concatenate(X).astype(np.float32); C = np.concatenate(C)
rec = S.build_record(X, np.zeros((len(X), 3), np.uint8), C,
                     np.ones(len(X), np.float32), np.zeros(len(X), np.uint8))
from builtin_interfaces.msg import Time as T
for _ in range(400):
    p.publish(make_cloud(rec, "map", T())); rclpy.spin_once(n, timeout_sec=0.05); time.sleep(0.4)
