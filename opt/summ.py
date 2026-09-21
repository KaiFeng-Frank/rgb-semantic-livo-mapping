import json,sys
d=json.load(open(sys.argv[1]))
K=['scans_received','scans_processed','scans_dropped_busy','scans_dropped_backpressure',
   'scans_no_pose','scans_published_est','map_voxels','wall_clock_s','peak_rss_gb',
   'points_mapped','frac_points_with_rgb','qdepth']
for k in K:
    if k in d and d[k] is not None: print('  %-28s %s'%(k,d[k]))
for k in ['ptv3_gpu_ms','ptv3_ms','stage_a_ms','wait_ms','gate_ms','deskew_ms','proj_ms',
          'map_ms','scanpub_ms','stage_b_ms','frame_period_ms','latency_ms','total_ms','snapshot_ms']:
    v=d.get(k)
    if v:
        g=lambda a: v.get(a, float('nan'))
        print('  %-16s mean %7.2f  p50 %7.2f  p95 %7.2f  max %8.2f'%(k,g('mean'),g('p50'),g('p95'),g('max')))
