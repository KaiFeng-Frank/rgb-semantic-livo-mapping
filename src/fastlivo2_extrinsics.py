#!/usr/bin/env python3
"""
Derive the FAST-LIVO2 extrinsic block for KITTI raw 2011_09_30 from KITTI's own
calibration files, and print the arithmetic.

FAST-LIVO2 conventions (READ FROM THE CODE, not assumed):

  LIVMapper.cpp:727   p_W = R_end * (extR * p_L + extT) + p_end
      -> extrin_calib.extrinsic_R / _T  ==  T_{IMU <- LiDAR}
      -> the state (_state.rot_end/pos_end, i.e. the evo TUM output) is T_{W <- IMU}

  vio.cpp:32-33  setImuToLidarExtrinsic(extT, extR):  Rli = extR^T ; Pli = -extR^T*extT
                 -> (Rli,Pli) = T_{L <- I}
  vio.cpp:58-59  Rci = Rcl * Rli ; Pci = Rcl * Pli + Pcl   ==>  T_{C<-I} = T_{C<-L} T_{L<-I}
      -> extrin_calib.Rcl / Pcl  ==  T_{camera <- LiDAR}   (p_cam = Rcl*p_velo + Pcl)

KITTI conventions:
  calib_velo_to_cam : p_cam0_unrect = R_vc * p_velo + T_vc          = T_{cam0u <- velo}
  calib_imu_to_velo : p_velo        = R_iv * p_imu  + T_iv          = T_{velo <- imu}
  calib_cam_to_cam  : y = P_rect_02 * R_rect_00 * T_{cam0u<-velo} * x_velo
                      P_rect_02 = K2 * [I | t2]  ->  t2 = K2^-1 * P_rect_02[:,3]
                      so  T_{rect2 <- cam0u} = (R_rect_00, t2)
"""
import numpy as np
np.set_printoptions(precision=9, suppress=False, linewidth=200)

CAL = "/data/livo_sem/data/raw/2011_09_30"

def read_calib(path):
    d = {}
    for line in open(path):
        if ":" not in line: continue
        k, v = line.split(":", 1)
        try: d[k.strip()] = np.array([float(x) for x in v.split()])
        except ValueError: pass
    return d

vc = read_calib(f"{CAL}/calib_velo_to_cam.txt")
iv = read_calib(f"{CAL}/calib_imu_to_velo.txt")
cc = read_calib(f"{CAL}/calib_cam_to_cam.txt")

def SE3(R, t):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t.ravel(); return T

# ---------------------------------------------------------------- 1. velo->cam0 (unrectified)
R_vc, T_vc = vc["R"].reshape(3,3), vc["T"]
T_cam0u_velo = SE3(R_vc, T_vc)

# ---------------------------------------------------------------- 2. cam0 unrect -> rect cam2
R_rect_00 = cc["R_rect_00"].reshape(3,3)
P_rect_02 = cc["P_rect_02"].reshape(3,4)
K2 = P_rect_02[:, :3]
t2 = np.linalg.solve(K2, P_rect_02[:, 3])          # K2^-1 * P[:,3]
T_rect2_cam0u = SE3(R_rect_00, t2)

# ---------------------------------------------------------------- 3. compose: cam2(rect) <- velo
T_cam2_velo = T_rect2_cam0u @ T_cam0u_velo
Rcl, Pcl = T_cam2_velo[:3,:3], T_cam2_velo[:3,3]

# ---------------------------------------------------------------- 4. imu <- lidar
R_iv, T_iv = iv["R"].reshape(3,3), iv["T"]
T_velo_imu = SE3(R_iv, T_iv)                        # T_{velo<-imu}
T_imu_velo = np.linalg.inv(T_velo_imu)              # T_{imu<-velo}  == FAST-LIVO2 extrinsic_R/T
extR, extT = T_imu_velo[:3,:3], T_imu_velo[:3,3]

# ---------------------------------------------------------------- report
print("="*78); print("STEP 1  T_{cam0_unrect <- velo}   (calib_velo_to_cam.txt, used verbatim)")
print(T_cam0u_velo)
print("\nSTEP 2  T_{cam2_rect <- cam0_unrect}")
print("  R = R_rect_00 =\n", R_rect_00)
print("  K2 = P_rect_02[:, :3] =\n", K2)
print("  P_rect_02[:,3] =", P_rect_02[:,3])
print("  t2 = K2^-1 * P_rect_02[:,3] =", t2, "  (|t2| = %.6f m, the cam0->cam2 stereo baseline)" % np.linalg.norm(t2))
print(T_rect2_cam0u)
print("\nSTEP 3  Rcl/Pcl = T_{cam2_rect <- velo} = T_{cam2<-cam0u} @ T_{cam0u<-velo}")
print(T_cam2_velo)
print("  Rcl (row-major, 9) =", ", ".join("%.9f" % v for v in Rcl.reshape(-1)))
print("  Pcl (3)            =", ", ".join("%.9f" % v for v in Pcl))
print("\nSTEP 4  extrinsic_R/T = T_{imu <- velo} = inv(T_{velo<-imu})   [calib_imu_to_velo.txt]")
print("  T_{velo<-imu} =\n", T_velo_imu)
print(T_imu_velo)
print("  extrinsic_R (row-major, 9) =", ", ".join("%.9f" % v for v in extR.reshape(-1)))
print("  extrinsic_T (3)            =", ", ".join("%.9f" % v for v in extT))

print("\n" + "="*78); print("SANITY CHECKS")
T_velo_cam2 = np.linalg.inv(T_cam2_velo)
t_L_C = T_velo_cam2[:3,3]
print("  camera origin expressed in the VELODYNE frame (x fwd, y left, z up):")
print("    t_{L<-C} = [%+.4f, %+.4f, %+.4f] m" % tuple(t_L_C))
print("    expected ~ [+0.27 fwd, +0.06 left, -0.08 down]  (velo 1.73 m, cam 1.65 m above ground)")
ok = (0.15 < t_L_C[0] < 0.40) and (-0.02 < t_L_C[1] < 0.15) and (-0.20 < t_L_C[2] < 0.02)
print("    VERDICT:", "PASS" if ok else "FAIL")
print("  IMU origin expressed in the VELODYNE frame:")
print("    t_{L<-I} = [%+.4f, %+.4f, %+.4f] m  (OXTS box is behind/below/right of the LiDAR)" % tuple(T_velo_imu[:3,3]))
print("  LiDAR origin expressed in the IMU frame (= extrinsic_T):")
print("    [%+.4f, %+.4f, %+.4f] m" % tuple(extT))
for n, R in (("Rcl", Rcl), ("extR", extR)):
    print("  orthonormality  %s: max|R R^T - I| = %.3e , det = %.9f"
          % (n, np.abs(R@R.T - np.eye(3)).max(), np.linalg.det(R)))

print("\n" + "="*78); print("CAMERA INTRINSICS for the RECTIFIED image_02 (what the bag publishes)")
print("  S_rect_02 = %d x %d" % (int(cc['S_rect_02'][0]), int(cc['S_rect_02'][1])))
print("  fx = %.6f  fy = %.6f  cx = %.6f  cy = %.6f" % (K2[0,0], K2[1,1], K2[0,2], K2[1,2]))
print("  distortion = 0,0,0,0 (image_02 of a _sync drive is already rectified/cropped)")
print("  (raw, UNRECTIFIED K_02 would be fx=%.4f fy=%.4f cx=%.4f cy=%.4f with D_02=%s"
      % (cc['K_02'].reshape(3,3)[0,0], cc['K_02'].reshape(3,3)[1,1],
         cc['K_02'].reshape(3,3)[0,2], cc['K_02'].reshape(3,3)[1,2], cc['D_02']))
print("   -- NOT used here.)")

np.save("/data/livo_sem/out/kitti_extrinsics.npy",
        {"T_cam2_velo": T_cam2_velo, "T_imu_velo": T_imu_velo,
         "T_velo_imu": T_velo_imu, "K2": K2, "S_rect_02": cc["S_rect_02"]},
        allow_pickle=True)
