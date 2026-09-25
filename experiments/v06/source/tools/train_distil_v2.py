#!/usr/bin/env python3
"""
train_distil_v2.py -- tools/train_distil_rprime.py plus `import split_v2`, which
registers the v0.6 split (seq 09 held out alongside seq 07).

ONE entry point covers all three v0.6 arms.  The v0.4/v0.5 entry points
(train_distil.py, train_distil_rprime.py) are left bit-identical so the arms trained
through them stay reproducible; the v0.6 arms only need to be consistent with each
other, so they share this file.

  python tools/train_distil_v2.py \
      --config-file src/Pointcept_v151/configs/semantic_kitti/arm_B0_v2_s1.py \
      --options save_path=/data/livo_sem/exp/sk2/armB0_s1
"""
import sys
sys.path.insert(0, "/data/livo_sem/src")
import pointcept_ext      # noqa: F401  MUST be first: installs the pointops stubs
import distil_ext         # noqa: F401  v0.4 model / dataset / hook
import distil_ext_rprime  # noqa: F401  arm Rprime's dataset and transform
import split_v2           # noqa: F401  v0.6 split subclasses

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
