#!/usr/bin/env python3
"""Does SemanticKITTI id 31/253 (bicyclist) include the BICYCLE, or only the human?
Decides nothing about the mapping (both land in two_wheeler either way) but it is the
evidence behind the claim that `person` stays clean.  CPU numpy only."""
import glob, numpy as np
S = sorted(glob.glob("/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/*.bin"))
L = sorted(glob.glob("/data/livo_sem/data/odometry/dataset/sequences/07/labels/*.label"))
zmins, zmaxs, heights, nearest_bike, nearest_person, nf = [], [], [], [], [], 0
for i in range(1101):
    g = np.fromfile(L[i], np.uint32) & 0xFFFF
    m = (g == 253) | (g == 31)
    if m.sum() < 30:
        continue
    nf += 1
    p = np.fromfile(S[i], np.float32).reshape(-1, 4)[:, :3]
    r = p[m]
    # local ground from road points in the same neighbourhood
    road = p[(g == 40)]
    c = r.mean(0)
    near_road = road[np.linalg.norm(road[:, :2] - c[:2], axis=1) < 5.0]
    gz = np.median(near_road[:, 2]) if len(near_road) > 20 else np.nan
    zmins.append(r[:, 2].min() - gz); zmaxs.append(r[:, 2].max() - gz)
    heights.append(r[:, 2].max() - r[:, 2].min())
    for lbl, acc in ((11, nearest_bike), (30, nearest_person)):
        q = p[g == lbl]
        if len(q):
            d = np.linalg.norm(q[:, None, :] - r[None, :, :], axis=2).min()
            acc.append(d)
print("frames with >=30 bicyclist points: %d" % nf)
print("bicyclist cluster, height above local road surface:")
print("   z_min  median %+.2f m   [%.2f .. %.2f]" % (np.nanmedian(zmins), np.nanmin(zmins), np.nanmax(zmins)))
print("   z_max  median %+.2f m   [%.2f .. %.2f]" % (np.nanmedian(zmaxs), np.nanmin(zmaxs), np.nanmax(zmaxs)))
print("   extent median %.2f m" % np.median(heights))
print("nearest SEPARATE bicycle(11) point to a bicyclist cluster: median %.2f m (n=%d)"
      % (np.median(nearest_bike), len(nearest_bike)) if nearest_bike else "no id-11 points in those frames")
print("nearest SEPARATE person(30)  point to a bicyclist cluster: median %.2f m (n=%d)"
      % (np.median(nearest_person), len(nearest_person)) if nearest_person else "no id-30 points in those frames")
