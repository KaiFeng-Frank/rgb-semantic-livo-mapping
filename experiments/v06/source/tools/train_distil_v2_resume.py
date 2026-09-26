#!/usr/bin/env python3
"""
train_distil_v2_resume.py -- tools/train_distil_v2.py with a SAFE `--resume-from`.

tools/train_distil_v2.py is unchanged and stays the entry point for fresh runs
(tools/train_distil_v2.py.pre_resume is a byte copy taken before this file existed).

  python tools/train_distil_v2_resume.py \
      --config-file src/Pointcept_v151/configs/semantic_kitti/arm_B0_noKL_v2_s2.py \
      --resume-from /data/livo_sem/exp/sk2/armB0_noKL_s2/model/model_last.pth \
      --options save_path=/data/livo_sem/exp/sk2/armB0_noKL_s2

NEVER RESUME A v0.6 ARM THROUGH POINTCEPT (`--options resume=True weight=...`)
------------------------------------------------------------------------------
CRITICAL_CONSTRAINTS.md T1.  Pointcept v1.5.1 CheckpointLoader.before_train
(pointcept/engines/hooks/misc.py:227-236):

        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                if comm.get_world_size() > 1:
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            ...
            if comm.get_world_size() == 1:
                key = key[7:]  # module.xxx.xxx -> xxx.xxx

On ONE GPU the prefix is never added, yet 7 characters are cut off every key, and the
result is loaded with strict=False.  A DistilSegmentorMiB checkpoint has no "module."
prefix, so frozen_backbone.X -> backbone.X (the frozen fp16 anchor, i.e. the RELEASED
weights, overwrite the student trunk), backbone.X -> e.X, seg_head.X -> d.X,
frozen_head.X -> head.X (all dropped as unexpected), 477 keys missing.  Epoch, best
metric, optimizer, scheduler and scaler are then restored correctly, so the run looks
resumed while it trains the released model.  Job 5 (armB0_noKL_s2) epoch 10, 2026-09-25.

WHAT THIS FILE DOES INSTEAD -- all of it before the first before_train hook runs
-------------------------------------------------------------------------------
  1. cfg.resume = False, cfg.weight = None: CheckpointLoader loads nothing (it only logs
     "No weight found at: None").
  2. The trainer is built by TRAINERS.build exactly as train_distil_v2.py builds it.  Its
     __init__ creates model, writer, loaders, optimizer, scheduler, scaler and hooks; nothing
     trains before .train().  restore_training_state() runs in between, so it precedes every
     before_train hook (CheckpointLoader is the first of them) and InformationWriter sees
     the restored start_epoch.  It refuses to run on a trainer that has started (EventStorage
     exists) or whose cfg would still let CheckpointLoader load.
  3. Model: torch.load(map_location="cpu", weights_only=False)["state_dict"]; a "module."
     prefix is removed ONLY when every key carries it (a mix is refused; nothing is ever cut by
     length); load_state_dict(strict=True); then every tensor of the model's state_dict is
     checked torch.equal to the checkpoint tensor of the SAME name.  strict=True alone never
     looks at values, and BatchNorm fills an absent num_batches_tracked silently even under
     strict=True.
  4. Optimizer / scheduler / scaler (if enable_amp) state dicts, after structural checks:
     equal param-group sizes, every saved moment has its parameter's shape, no moments on a
     parameter that is frozen now, scheduler total_steps equal to the rebuilt one.  The
     restored optimizer state is compared back to the checkpoint.  Then
     start_epoch = checkpoint["epoch"], best_metric_value = checkpoint["best_metric_value"].
  5. KL arms: kl_lambda is a plain attribute calibrated at step 0 of the original run and is
     NOT in the checkpoint; left alone it would be re-calibrated on the TRAINED student at
     the first resumed step.  It is restored from <run>/rare_class_audit.jsonl (written every
     epoch with the exact float repr), or the resume is refused.
  6. Side effects as in Pointcept's own resume: train.log is appended to, not truncated
     (Trainer.__init__ opens it with "w" when cfg.resume is False, so the root logger is
     created first, in append mode), and config.py is not re-dumped.  The current config must
     equal the run's config.py (save_path line aside) or the resume is refused.

tools/verify_resume_restore.py exercises this on CPU against real checkpoints.
"""
import sys
sys.path.insert(0, "/data/livo_sem/src")
import pointcept_ext      # noqa: F401  MUST be first: installs the pointops stubs
import distil_ext         # noqa: F401  v0.4 model / dataset / hook
import distil_ext_rprime  # noqa: F401  arm Rprime's dataset and transform
import split_v2           # noqa: F401  v0.6 split subclasses

import os
import copy
import json
import difflib
from collections import OrderedDict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from pointcept.engines.defaults import (
    default_argument_parser, default_config_parser, default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from pointcept.engines.hooks.misc import CheckpointLoader
from pointcept.utils.logger import get_root_logger

MODULE_PREFIX = "module."


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


# ----------------------------------------------------------------------------- model
def strip_genuine_module_prefix(state_dict):
    """(state_dict, n_stripped).  "module." is removed only when EVERY key starts with it
    (a DDP save).  No prefix anywhere: the same object comes back, _metadata intact.  A mix
    is refused.  Nothing is ever cut by a fixed number of characters (that is T1)."""
    keys = list(state_dict.keys())
    n = sum(k.startswith(MODULE_PREFIX) for k in keys)
    if n == 0:
        return state_dict, 0
    if n != len(keys):
        raise RuntimeError(
            "checkpoint state_dict mixes %d 'module.'-prefixed keys with %d unprefixed "
            "ones; refusing to guess which are real" % (n, len(keys) - n))
    sd = OrderedDict(state_dict)
    meta = getattr(state_dict, "_metadata", None)
    if meta is not None:
        sd._metadata = copy.deepcopy(meta)
    consume_prefix_in_state_dict_if_present(sd, MODULE_PREFIX)   # startswith-guarded
    return sd, n


def restore_model(model, ckpt_state_dict):
    target = unwrap(model)
    sd, n_stripped = strip_genuine_module_prefix(ckpt_state_dict)
    info = target.load_state_dict(sd, strict=True)   # raises on missing/unexpected/shape
    missing, unexpected = list(info.missing_keys), list(info.unexpected_keys)
    if missing or unexpected:          # unreachable under strict=True; kept as the record
        raise RuntimeError("strict load reported %d missing / %d unexpected keys"
                           % (len(missing), len(unexpected)))
    cur = target.state_dict()
    absent = [k for k in cur if k not in sd]
    extra = [k for k in sd if k not in cur]
    differ, cast = [], 0
    for k, v in cur.items():
        if k not in sd:
            continue
        ref = sd[k]
        if ref.dtype != v.dtype:
            cast += 1
        if not torch.equal(v.detach().cpu(), ref.to(dtype=v.dtype)):
            differ.append(k)
    if absent or extra or differ:
        raise RuntimeError(
            "model restore self-check FAILED: %d model tensors absent from the checkpoint, "
            "%d checkpoint tensors not in the model, %d tensors differ; first: %s"
            % (len(absent), len(extra), len(differ), (absent + extra + differ)[:5]))
    return dict(tensors=len(cur), missing=len(missing), unexpected=len(unexpected),
                prefix_stripped=n_stripped, dtype_cast=cast)


# ------------------------------------------------------------------ optimizer / sched
def check_optimizer_compat(optimizer, saved):
    groups, sgroups = optimizer.param_groups, saved["param_groups"]
    if len(groups) != len(sgroups):
        raise RuntimeError("optimizer: %d param groups now, %d in the checkpoint"
                           % (len(groups), len(sgroups)))
    for i, (g, sg) in enumerate(zip(groups, sgroups)):
        if len(g["params"]) != len(sg["params"]):
            raise RuntimeError("optimizer group %d: %d params now, %d in the checkpoint"
                               % (i, len(g["params"]), len(sg["params"])))
    # the positional pairing torch's Optimizer.load_state_dict itself uses
    id_map = dict(zip(chain.from_iterable(sg["params"] for sg in sgroups),
                      chain.from_iterable(g["params"] for g in groups)))
    bad = []
    for idx, st in saved["state"].items():
        p = id_map.get(idx)
        if p is None:
            bad.append((idx, "no parameter at this position"))
            continue
        if not p.requires_grad:
            bad.append((idx, "moments saved for a parameter that is frozen now"))
        for key, t in st.items():
            if key != "step" and torch.is_tensor(t) and tuple(t.shape) != tuple(p.shape):
                bad.append((idx, "%s %s != parameter %s"
                            % (key, tuple(t.shape), tuple(p.shape))))
    if bad:
        raise RuntimeError("optimizer state does not fit the rebuilt parameters: %s" % bad[:5])
    return dict(group_sizes=[len(g["params"]) for g in groups],
                state_entries=len(saved["state"]),
                trainable=sum(1 for g in groups for p in g["params"] if p.requires_grad))


def _same(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape
                and torch.equal(a.detach().cpu(), b.detach().cpu().to(a.dtype)))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def check_optimizer_restored(optimizer, saved):
    now = optimizer.state_dict()
    if not _same(now["param_groups"], saved["param_groups"]):
        raise RuntimeError("optimizer param_groups differ from the checkpoint after load")
    if now["state"].keys() != saved["state"].keys():
        raise RuntimeError("optimizer state indices differ from the checkpoint after load")
    bad = [i for i in saved["state"] if not _same(now["state"][i], saved["state"][i])]
    if bad:
        raise RuntimeError("optimizer state differs from the checkpoint after load: %s" % bad[:5])
    steps = sorted({float(st["step"]) for st in saved["state"].values() if "step" in st})
    return dict(restored_equal=True, steps=steps)


def check_scheduler_compat(scheduler, saved):
    now, then = getattr(scheduler, "total_steps", None), saved.get("total_steps")
    if now is not None and then is not None and now != then:
        raise RuntimeError(
            "scheduler total_steps: %d rebuilt from this config and loader, %d in the "
            "checkpoint -- the epoch length or eval_epoch changed; refusing to splice two "
            "schedules" % (now, then))


# ------------------------------------------------------------------------- kl_lambda
def restore_kl_lambda(model, run_dir, ckpt_epoch):
    m = unwrap(model)
    if not getattr(m, "kl_enabled", False):
        return "not needed (kl_enabled=False)"
    if m.kl_lambda is not None:
        return "not needed (pinned by the config: %r)" % m.kl_lambda
    audit = os.path.join(run_dir, "rare_class_audit.jsonl")
    if not os.path.isfile(audit):
        raise RuntimeError(
            "refusing to resume a KL arm: kl_lambda was calibrated at step 0, is not in the "
            "checkpoint, and %s (which records it every epoch) is missing" % audit)
    vals = []
    with open(audit) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if int(rec["epoch"]) <= ckpt_epoch:
                    vals.append(rec.get("kl_lambda"))
    if not vals or any(v is None for v in vals) or len(set(vals)) != 1:
        raise RuntimeError(
            "refusing to resume a KL arm: %s gives kl_lambda %s for epochs <= %d; "
            "exactly one non-null value is required"
            % (audit, sorted(set(map(repr, vals))), ckpt_epoch))
    m.kl_lambda = float(vals[0])
    return "restored %r from %s (%d epoch records <= %d, all identical)" % (
        m.kl_lambda, audit, len(vals), ckpt_epoch)


# ---------------------------------------------------------------------------- config
def check_config_matches_run(cfg, run_dir):
    path = os.path.join(run_dir, "config.py")
    if not os.path.isfile(path):
        raise RuntimeError("refusing to resume: %s (the config this run started with) "
                           "is missing" % path)
    with open(path) as fh:
        saved = fh.read().splitlines()
    now = cfg.pretty_text.splitlines()

    def drop(lines):
        return [ln for ln in lines if not ln.startswith("save_path = ")]

    if drop(saved) != drop(now):
        diff = "\n".join(difflib.unified_diff(drop(saved), drop(now), path,
                                              "current config", lineterm="", n=1))
        raise RuntimeError("refusing to resume: the current config differs from the one "
                           "this run was started with:\n" + diff)
    return "identical to %s (%d lines; the save_path line is not compared)" % (path, len(saved))


# --------------------------------------------------------------------------- restore
def restore_training_state(trainer, ckpt_path, run_dir):
    """Restore a built, not-yet-started trainer from ckpt_path.  Returns a report dict."""
    for attr in ("model", "optimizer", "scheduler", "scaler", "hooks", "train_loader"):
        if not hasattr(trainer, attr):
            raise RuntimeError("restore called before the trainer built `%s`" % attr)
    if hasattr(trainer, "storage"):
        raise RuntimeError("restore called after training started (EventStorage exists)")
    if trainer.cfg.weight or trainer.cfg.resume:
        raise RuntimeError("cfg.weight / cfg.resume are set: Pointcept's CheckpointLoader "
                           "would load on top of this restore (CRITICAL_CONSTRAINTS.md T1)")
    log = trainer.logger
    ck = load_checkpoint(ckpt_path)
    rep = dict(checkpoint=ckpt_path, epoch=int(ck["epoch"]),
               best_metric_value=ck["best_metric_value"],
               checkpoint_loader_hooks=sum(isinstance(h, CheckpointLoader)
                                           for h in trainer.hooks))
    rep["model"] = restore_model(trainer.model, ck["state_dict"])
    rep["optimizer"] = check_optimizer_compat(trainer.optimizer, ck["optimizer"])
    trainer.optimizer.load_state_dict(ck["optimizer"])
    rep["optimizer"].update(check_optimizer_restored(trainer.optimizer, ck["optimizer"]))
    check_scheduler_compat(trainer.scheduler, ck["scheduler"])
    trainer.scheduler.load_state_dict(ck["scheduler"])
    rep["scheduler"] = dict(last_epoch=trainer.scheduler.last_epoch,
                            total_steps=getattr(trainer.scheduler, "total_steps", None))
    if trainer.cfg.enable_amp:
        if trainer.scaler is None or ck.get("scaler") is None:
            raise RuntimeError("enable_amp is set but the %s has no scaler state"
                               % ("trainer" if trainer.scaler is None else "checkpoint"))
        trainer.scaler.load_state_dict(ck["scaler"])
        rep["scaler"] = dict(ck["scaler"])
    else:
        rep["scaler"] = None
    trainer.start_epoch = ck["epoch"]
    trainer.best_metric_value = ck["best_metric_value"]
    rep["kl_lambda"] = restore_kl_lambda(trainer.model, run_dir, rep["epoch"])
    del ck

    m, o = rep["model"], rep["optimizer"]
    log.info("=> RESUME model: strict=True, %d tensors, %d missing, %d unexpected, "
             "every tensor torch.equal to the same-name checkpoint tensor "
             "('module.' stripped from %d keys, %d dtype casts)"
             % (m["tensors"], m["missing"], m["unexpected"], m["prefix_stripped"],
                m["dtype_cast"]))
    log.info("=> RESUME optimizer: groups %s, %d moment entries (%d trainable params), "
             "step %s, restored state equal to the checkpoint"
             % (o["group_sizes"], o["state_entries"], o["trainable"], o["steps"]))
    log.info("=> RESUME scheduler: last_epoch %s / total_steps %s;  scaler: %s"
             % (rep["scheduler"]["last_epoch"], rep["scheduler"]["total_steps"], rep["scaler"]))
    log.info("=> RESUME start_epoch %d, best_metric_value %r;  kl_lambda: %s;  "
             "CheckpointLoader hooks present: %d (cfg.weight=None -> they load nothing)"
             % (trainer.start_epoch, trainer.best_metric_value, rep["kl_lambda"],
                rep["checkpoint_loader_hooks"]))
    if trainer.start_epoch >= trainer.max_epoch:
        log.warning("=> RESUME checkpoint is at the final epoch (%d/%d): nothing left to train"
                    % (trainer.start_epoch, trainer.max_epoch))
    return rep


def default_trainer_builder(cfg):
    return TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))


def build_resumed_trainer(cfg, resume_from, trainer_builder=default_trainer_builder,
                          run_dir=None, require_same_save_path=True):
    """Everything main_worker does except .train().  Returns (trainer, report).
    trainer_builder and the two run_dir arguments exist for tools/verify_resume_restore.py
    (a CPU trainer and a scratch save_path); the training path uses the defaults."""
    ckpt_path = os.path.realpath(resume_from)
    run_dir = os.path.realpath(run_dir or os.path.dirname(os.path.dirname(ckpt_path)))
    if require_same_save_path and os.path.realpath(cfg.save_path) != run_dir:
        raise RuntimeError("save_path %s is not the run %s that %s belongs to"
                           % (cfg.save_path, run_dir, resume_from))
    # 1. Pointcept's CheckpointLoader must load nothing
    cfg.resume = False
    cfg.weight = None
    config_note = check_config_matches_run(cfg, run_dir)   # before default_setup adds keys
    cfg = default_setup(cfg)
    # 6. Trainer.__init__ opens train.log with "w" when cfg.resume is False: create the root
    #    logger first, in append mode, so the run's log is continued and not truncated.
    logger = get_root_logger(log_file=os.path.join(cfg.save_path, "train.log"), file_mode="a")
    logger.info("=> RESUME via tools/train_distil_v2_resume.py (CRITICAL_CONSTRAINTS.md T1) "
                "from %s" % ckpt_path)
    logger.info("=> RESUME config: %s" % config_note)
    # 2. built exactly as train_distil_v2.py builds it; nothing has trained yet
    trainer = trainer_builder(cfg)
    report = restore_training_state(trainer, ckpt_path, run_dir)
    report["config"] = config_note
    return trainer, report


def main_worker(cfg, resume_from):
    trainer, _ = build_resumed_trainer(cfg, resume_from)
    trainer.train()


def main():
    parser = default_argument_parser()
    parser.add_argument("--resume-from", required=True, metavar="MODEL_LAST_PTH",
                        help="model_last.pth of THIS run (the checkpoint's run dir must be "
                             "save_path)")
    args = parser.parse_args()
    assert args.num_gpus == 1 and args.num_machines == 1, "single-process only"
    opts = dict(args.options or {})
    bad = sorted(k for k in opts if k in ("resume", "weight"))
    if bad:
        parser.error("--resume-from replaces %s: do not pass them (CRITICAL_CONSTRAINTS.md T1)"
                     % ", ".join(bad))
    if not os.path.isfile(args.resume_from):
        parser.error("no such checkpoint: %s" % args.resume_from)
    # resume=True here ONLY stops default_config_parser from re-dumping config.py over the
    # copy the original run wrote (engines/defaults.py:126), as Pointcept's own resume does.
    # build_resumed_trainer sets it back to False before anything else reads it.
    opts["resume"] = True
    cfg = default_config_parser(args.config_file, opts)
    launch(main_worker, num_gpus_per_machine=1, num_machines=1, machine_rank=0,
           dist_url=args.dist_url, cfg=(cfg, args.resume_from))


if __name__ == "__main__":
    main()
