#!/usr/bin/env python3
"""
train_distil.py -- tools/train_ptv3_sk.py plus `import distil_ext`, which registers
DistilSegmentorMiB, DistilSemanticKITTIDataset and the RareClassAudit hook.

scripts/train.sh is STILL not usable: it copies pointcept into exp/<n>/code and
re-imports an unstubbed copy.  Single process, no torchrun, no NCCL.

  python tools/train_distil.py \
      --config-file src/Pointcept_v151/configs/semantic_kitti/semseg-pt-v3m1-distil-common9.py \
      --options save_path=exp/sk/armB1 model.freeze_backbone=True
"""
import sys
sys.path.insert(0, "/data/wuyou/livo_sem/src")
import pointcept_ext   # noqa: F401  MUST be first: installs the pointops stubs
import distil_ext      # noqa: F401  registers the v0.4 model/dataset/hook

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
