#!/usr/bin/env python3
"""
train_distil_rprime.py -- tools/train_distil.py plus `import distil_ext_rprime`, which
registers DistilSemanticKITTIDatasetVR and the VoxelRandomSupervise transform.

It is a separate entry point rather than an edit to train_distil.py for the same reason
distil_ext_rprime.py is a separate module: arms B1 / D / B0 / C / R were trained through
train_distil.py at md5 a1f8b636c67c21ac9830bdc20f6aa313, and that file stays bit-identical
on this machine so the comparison arms remain reproducible from it.

  python tools/train_distil_rprime.py \
      --config-file src/Pointcept_v151/configs/semantic_kitti/arm_Rprime.py \
      --options save_path=exp/sk/armRprime
"""
import sys
sys.path.insert(0, "/data/wuyou/livo_sem/src")
import pointcept_ext      # noqa: F401  MUST be first: installs the pointops stubs
import distil_ext         # noqa: F401  registers the v0.4 model/dataset/hook
import distil_ext_rprime  # noqa: F401  registers arm R''s dataset and transform

from pointcept.engines.defaults import (
    default_argument_parser, default_config_parser, default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    TRAINERS.build(dict(type=cfg.train.type, cfg=cfg)).train()


def main():
    args = default_argument_parser().parse_args()
    assert args.num_gpus == 1 and args.num_machines == 1, "single-process only"
    cfg = default_config_parser(args.config_file, args.options)
    launch(main_worker, num_gpus_per_machine=1, num_machines=1,
           machine_rank=0, dist_url=args.dist_url, cfg=(cfg,))


if __name__ == "__main__":
    main()
