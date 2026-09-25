#!/usr/bin/env python3
"""
tools/v05_patch.py -- the ONLY source change v0.5 makes to the qualified pipeline:
a checkpoint selector, threaded node -> client -> worker -> loader.

Every replacement below is an EXACT-STRING substitution that must match exactly once;
the script aborts before writing anything if any anchor is missing or ambiguous, so
the edit is auditable against src/_pre_v05_backup/.  With no --ptv3-ckpt (no
PTV3_CKPT in the worker's environment) every path is byte-for-byte the v0.2/v0.3
behaviour: the released nuScenes checkpoint is loaded from the same constant.
"""
import sys

SRC = "/data/livo_sem/src/"

EDITS = {
    # ------------------------------------------------------------- loader
    "ptv3_loader_verified.py": [
        ('def build_ptv3(device="cuda", enable_flash=None, shuffle_orders=None, half=False):',
         'def build_ptv3(device="cuda", enable_flash=None, shuffle_orders=None, half=False,\n'
         '               ckpt_path=None):'),
        ('    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)\n'
         '    sd = sd.get("state_dict", sd)\n'
         '    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}\n'
         '    model.load_state_dict(sd, strict=True)      # MUST stay strict\n',
         '    # v0.5: ckpt_path selects a checkpoint that carries the SAME 488-tensor\n'
         '    # student structure (tools/extract_student.py writes them).  Default = the\n'
         '    # released nuScenes checkpoint, exactly as before.  strict=True either way.\n'
         '    path = ckpt_path or CKPT_PATH\n'
         '    sd = torch.load(path, map_location="cpu", weights_only=False)\n'
         '    sd = sd.get("state_dict", sd)\n'
         '    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}\n'
         '    model.load_state_dict(sd, strict=True)      # MUST stay strict\n'
         '    model._ckpt_path = path\n'
         '    model._n_ckpt_tensors = len(sd)\n'),
    ],
    # ------------------------------------------------------------- worker
    "ptv3_worker.py": [
        ('FAST_HILBERT = int(os.environ.get("PTV3_FAST_HILBERT", "1")) != 0\n',
         'FAST_HILBERT = int(os.environ.get("PTV3_FAST_HILBERT", "1")) != 0\n'
         'CKPT = os.environ.get("PTV3_CKPT", "").strip() or None   # v0.5: None = released\n'),
        ('                 fast_voxel=FAST_VOXEL, gpu_voxel=GPU_VOXEL,\n'
         '                 fast_hilbert=FAST_HILBERT):\n',
         '                 fast_voxel=FAST_VOXEL, gpu_voxel=GPU_VOXEL,\n'
         '                 fast_hilbert=FAST_HILBERT, ckpt=CKPT):\n'),
        ('        self.model, self.cfg = build_ptv3(device=device, shuffle_orders=bool(shuffle),\n'
         '                                          half=bool(half))\n',
         '        self.model, self.cfg = build_ptv3(device=device, shuffle_orders=bool(shuffle),\n'
         '                                          half=bool(half), ckpt_path=ckpt)\n'
         '        self.ckpt = getattr(self.model, "_ckpt_path", None)\n'
         '        self.ckpt_tensors = int(getattr(self.model, "_n_ckpt_tensors", 0))\n'
         '        try:\n'
         '            import hashlib\n'
         '            with open(self.ckpt, "rb") as _f:\n'
         '                self.ckpt_sha256 = hashlib.sha256(_f.read()).hexdigest()\n'
         '        except Exception:\n'
         '            self.ckpt_sha256 = "?"\n'),
        ('    sys.stderr.write("[ptv3_worker] gpu_voxel=%s fast_hilbert=%s\\n"\n'
         '                     % (seg.gpu_voxel, seg.fast_hilbert))\n',
         '    sys.stderr.write("[ptv3_worker] gpu_voxel=%s fast_hilbert=%s\\n"\n'
         '                     % (seg.gpu_voxel, seg.fast_hilbert))\n'
         '    sys.stderr.write("[ptv3_worker] checkpoint=%s tensors=%d sha256=%s\\n"\n'
         '                     % (seg.ckpt, seg.ckpt_tensors, seg.ckpt_sha256))\n'),
    ],
    # ------------------------------------------------------------- client
    "ptv3_client.py": [
        ('                 half=None, shuffle=None, fast_voxel=None, tf32=None):\n',
         '                 half=None, shuffle=None, fast_voxel=None, tf32=None,\n'
         '                 ckpt=None):\n'),
        ('        # keep the conda env\'s own libs first; strip ROS python paths\n'
         '        env.pop("PYTHONPATH", None)\n',
         '        if ckpt:\n'
         '            env["PTV3_CKPT"] = str(ckpt)          # v0.5 checkpoint selector\n'
         '        else:\n'
         '            env.pop("PTV3_CKPT", None)\n'
         '        # keep the conda env\'s own libs first; strip ROS python paths\n'
         '        env.pop("PYTHONPATH", None)\n'),
    ],
    # ------------------------------------------------------------- node
    "semantic_map_node.py": [
        ('                               fast_voxel=args.fast_voxel, tf32=args.tf32)\n'
         '        self.get_logger().info("PTv3 worker ready (%d classes), shm=%s pipeline=%s"\n'
         '                               % (self.ptv3.num_classes, not args.no_shm,\n'
         '                                  not args.no_pipeline))\n',
         '                               fast_voxel=args.fast_voxel, tf32=args.tf32,\n'
         '                               ckpt=args.ptv3_ckpt)\n'
         '        self.get_logger().info("PTv3 worker ready (%d classes), shm=%s pipeline=%s ckpt=%s"\n'
         '                               % (self.ptv3.num_classes, not args.no_shm,\n'
         '                                  not args.no_pipeline, args.ptv3_ckpt or "released"))\n'),
        ('            ptv3_grid=self.args.grid_size, ptv3_fast_voxel=self.args.fast_voxel,\n',
         '            ptv3_grid=self.args.grid_size, ptv3_fast_voxel=self.args.fast_voxel,\n'
         '            ptv3_ckpt=self.args.ptv3_ckpt,\n'),
        ('    ap.add_argument("--ptv3-log", default="/data/livo_sem/logs/ptv3_worker.log")\n',
         '    ap.add_argument("--ptv3-log", default="/data/livo_sem/logs/ptv3_worker.log")\n'
         '    ap.add_argument("--ptv3-ckpt", default=None,\n'
         '                    help="v0.5: a 488-tensor student checkpoint written by "\n'
         '                         "tools/extract_student.py; default = the released nuScenes weights")\n'),
    ],
}


def main():
    dry = "--dry" in sys.argv
    plan = []
    for fn, edits in EDITS.items():
        p = SRC + fn
        txt = open(p).read()
        for old, new in edits:
            c = txt.count(old)
            if c != 1:
                print("ABORT: anchor matched %d times in %s:\n%s" % (c, fn, old))
                sys.exit(2)
            txt = txt.replace(old, new)
        plan.append((p, txt, len(edits)))
    if dry:
        print("dry run: all anchors matched exactly once")
        return
    for p, txt, n in plan:
        open(p, "w").write(txt)
        print("patched %s (%d edits)" % (p, n))


if __name__ == "__main__":
    main()
