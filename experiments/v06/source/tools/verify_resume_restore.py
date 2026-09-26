#!/usr/bin/env python3
"""
verify_resume_restore.py -- CPU proof that tools/train_distil_v2_resume.py restores a run
exactly, and (negative control) that Pointcept v1.5.1's own resume path does not.

  CUDA_VISIBLE_DEVICES="" python tools/verify_resume_restore.py --path fixed \
      --config src/Pointcept_v151/configs/semantic_kitti/arm_B0_noKL_v2_s2.py \
      --ckpt exp/sk2/armB0_noKL_s2/model/model_last.pth \
      --out out/v06_train/resume_restore_verify.txt

--path fixed      R.build_resumed_trainer() -- the very function the resume tool's
                  main_worker calls -- then the hooks' before_train exactly as
                  Trainer.train() runs it, then the checks.
--path pointcept  Pointcept's own path, as opt/train_queue_v06_resume.sh.pre_fix drove it:
                  options resume=True weight=<ckpt>, trainer built, hooks' before_train
                  (CheckpointLoader first).  The ONLY change to Pointcept's code path: its
                  torch.load gets map_location="cpu" instead of storage.cuda().

The trainer is the config's own trainer class with two CPU substitutions and nothing else:
build_model() without .cuda() (fp32 student; the fp16 anchor as the model builds it), and
build_scaler() -> torch.amp.GradScaler("cpu"), because the CUDA scaler disables itself
without a GPU and would silently load nothing.  Everything it writes (train.log, tensorboard
events) goes to a scratch save_path; the run directory is only read.

The checks are taken AFTER before_train -- the instant the first training step would start --
against an INDEPENDENT torch.load of the checkpoint (on CPU, Optimizer.load_state_dict keeps
references to same-device tensors, so comparing with the dict that was loaded would compare
tensors with themselves):
  (a) the model load: strict flag, missing / unexpected key counts
  (b) every tensor of model.state_dict() torch.equal to the checkpoint tensor of the SAME name
      (dtype matched).  Counter-check: how many backbone.* tensors of the checkpoint differ
      from frozen_backbone.* (student rounded to the anchor's fp16 -- an untrained student
      would count 0), and whether the model's trunk is the checkpoint's backbone.* or its
      frozen_backbone.*
  (c) optimizer state_dict == checkpoint (param_groups, step, exp_avg, exp_avg_sq); also the
      scheduler and scaler state_dicts
  (d) start_epoch == checkpoint["epoch"], best_metric_value == checkpoint[...], and
      InformationWriter's iteration counter == start_epoch * len(train_loader) (the hooks ran
      after the restore and saw it)
Exit status 0 = the expected outcome for the path (fixed: all pass; pointcept: the defect
reproduced), 1 otherwise.
"""
import sys
import os
import time
import socket
import hashlib
import argparse
import json
import io
import copy
import contextlib
import datetime
import resource
import collections

sys.path.insert(0, "/data/livo_sem/tools")
sys.path.insert(0, "/data/livo_sem/src")
import train_distil_v2_resume as R    # noqa: E402  imports pointcept_ext first, as the tool does
import torch                          # noqa: E402
import distil_ext as DX               # noqa: E402
from pointcept.engines.defaults import (default_config_parser, default_setup,  # noqa: E402
                                        create_ddp_model)
from pointcept.engines.train import TRAINERS                                   # noqa: E402
from pointcept.engines.hooks.misc import InformationWriter                     # noqa: E402
from pointcept.models import build_model                                       # noqa: E402
from pointcept.utils.events import EventStorage                                # noqa: E402


class Tee:
    def __init__(self, path):
        self.path, self.lines = path, []

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(s)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            fh.write("\n".join(self.lines) + "\n")


def cpu_trainer_class(cfg):
    base = TRAINERS.get(cfg.train.type)

    class CPUTrainer(base):
        """The config's trainer with two CPU substitutions and nothing else."""

        def build_model(self):
            model = build_model(self.cfg.model)
            if self.cfg.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Num params: {n}")
            return create_ddp_model(model, broadcast_buffers=False,     # no .cuda()
                                    find_unused_parameters=self.cfg.find_unused_parameters)

        def build_scaler(self):
            return torch.amp.GradScaler("cpu") if self.cfg.enable_amp else None

    CPUTrainer.__name__ = "CPU" + base.__name__
    return CPUTrainer


def sha256(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(bs), b""):
            h.update(blk)
    return h.hexdigest()


def top(k):
    return k.split(".")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", choices=("fixed", "pointcept", "guards"), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/data/livo_sem/out/v06_train/resume_verify_work")
    ap.add_argument("--stated-epoch", type=int, default=None,
                    help="epoch the request says the file holds (reported, not used)")
    ap.add_argument("--stated-best", type=float, default=None,
                    help="best_metric_value the request says the file holds (reported)")
    ap.add_argument("--expect-missing", type=int, default=477,
                    help="pointcept path: missing-key count the defect produced in the log")
    args = ap.parse_args()
    if torch.cuda.is_available():
        sys.exit('refusing to run with a visible GPU: set CUDA_VISIBLE_DEVICES=""')
    torch.set_num_threads(2)
    T = Tee(args.out)
    t0 = time.time()

    ckpt = os.path.realpath(args.ckpt)
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    st = os.stat(ckpt)
    save_path = os.path.join(args.scratch, "%s_%s" % (os.path.basename(run_dir), args.path))
    os.makedirs(save_path, exist_ok=True)
    for f in os.listdir(save_path):                  # this run's scratch log only
        if f == "train.log" or f.startswith("events.out.tfevents"):
            os.remove(os.path.join(save_path, f))

    T("verify_resume_restore.py --path %s        %s  host %s" % (
        args.path, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), socket.gethostname()))
    T("torch %s, CUDA visible: %s, threads %d" % (torch.__version__, torch.cuda.is_available(),
                                                  torch.get_num_threads()))
    T("scripts     tools/verify_resume_restore.py sha256 %s, tools/train_distil_v2_resume.py sha256 %s"
      % (sha256(os.path.abspath(__file__))[:16], sha256(R.__file__)[:16]))
    T("config      %s" % os.path.realpath(args.config))
    T("checkpoint  %s" % ckpt)
    T("            %d bytes, mtime %s, sha256 %s" % (
        st.st_size, datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S.%f"),
        sha256(ckpt)))
    T("run dir     %s   (read only)" % run_dir)
    T("scratch     %s   (train.log / events of this check)" % save_path)
    T("")

    if args.path == "guards":
        ok = run_guards(T, args, ckpt, run_dir, save_path)
        T("wall %.1f s, peak RSS %.2f GB" % (time.time() - t0,
                                              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
        T.save()
        return 0 if ok else 1

    # ------------------------------------------------------------ instrumentation (observe only)
    phase = ["build+restore"]
    calls = []

    def spy_on(trainer):
        orig = trainer.model.load_state_dict

        def load_state_dict(state_dict, strict=True, *a, **k):
            res = orig(state_dict, strict=strict, *a, **k)
            calls.append(dict(phase=phase[0], strict=strict, keys=list(state_dict.keys()),
                              missing=list(res.missing_keys),
                              unexpected=list(res.unexpected_keys)))
            return res
        trainer.model.load_state_dict = load_state_dict

    real_load = torch.load
    loads = []

    def cpu_load(f, *a, **k):              # CheckpointLoader passes storage.cuda()
        loads.append((phase[0], str(f)))
        k["map_location"] = "cpu"
        k.setdefault("weights_only", False)
        return real_load(f, *a, **k)

    # ------------------------------------------------------------ build (+ restore)
    if args.path == "fixed":
        cfg = default_config_parser(args.config, {"save_path": save_path, "resume": True})

        def builder(c):
            t = cpu_trainer_class(c)(c)
            spy_on(t)
            return t
        trainer, rep = R.build_resumed_trainer(cfg, ckpt, trainer_builder=builder,
                                               run_dir=run_dir, require_same_save_path=False)
    else:
        cfg = default_config_parser(args.config,
                                    {"save_path": save_path, "resume": True, "weight": ckpt})
        cfg = default_setup(cfg)
        trainer = cpu_trainer_class(cfg)(cfg)
        spy_on(trainer)
        rep = None
    started = hasattr(trainer, "storage")

    # ------------------------------------------------------------ before_train, as train() runs it
    phase[0] = "before_train"
    torch.load = cpu_load
    try:
        with EventStorage() as trainer.storage:
            trainer.before_train()
    finally:
        torch.load = real_load
    phase[0] = "checks"

    ref = torch.load(ckpt, map_location="cpu", weights_only=False)      # independent copy
    rsd = ref["state_dict"]
    model = R.unwrap(trainer.model)
    msd = model.state_dict()
    rel = DX._load_released_state_dict()
    T("checkpoint holds: epoch %r, best_metric_value %r, %d state_dict tensors %s" % (
        ref["epoch"], ref["best_metric_value"], len(rsd),
        dict(collections.Counter(top(k) for k in rsd))))
    if args.stated_epoch is not None or args.stated_best is not None:
        same = (args.stated_epoch == ref["epoch"] and args.stated_best == ref["best_metric_value"])
        T("request stated:   epoch %r, best_metric_value %r  -> %s" % (
            args.stated_epoch, args.stated_best,
            "matches this file" if same else "does NOT match this file (see note at the end)"))
    T("trainer class %s; restore ran before training started: %s; hooks: %s" % (
        type(trainer).__name__, not started, [type(h).__name__ for h in trainer.hooks]))
    T("torch.load calls during before_train: %d %s" % (
        len(loads), [os.path.basename(f) for _, f in loads]))
    T("")

    # ------------------------------------------------------------ (a)
    T("(a) model load")
    for c in calls:
        T("    %-13s load_state_dict(strict=%s): %d keys given -> %d missing, %d unexpected" % (
            c["phase"], c["strict"], len(c["keys"]), len(c["missing"]), len(c["unexpected"])))
    rc = [c for c in calls if c["phase"] == "build+restore"]
    hc = [c for c in calls if c["phase"] == "before_train"]
    if args.path == "fixed":
        a_ok = (len(rc) == 1 and rc[0]["strict"] is True and not rc[0]["missing"]
                and not rc[0]["unexpected"] and len(hc) == 0)
        T("    restore: strict=True, %d missing / %d unexpected; CheckpointLoader loaded %s"
          % (rep["model"]["missing"], rep["model"]["unexpected"],
             "nothing" if not hc else "%d time(s)!" % len(hc)))
        T("    'module.' stripped from %d keys, %d dtype casts" % (
            rep["model"]["prefix_stripped"], rep["model"]["dtype_cast"]))
    else:
        c = hc[0] if hc else None
        a_ok = False
        if c is not None:
            sliced = [k[7:] for k in rsd.keys()]
            T("    hook fed the model [k[7:] for k in checkpoint keys]: %s" % (c["keys"] == sliced))
            for k in ("frozen_backbone.embedding.stem.conv.weight",
                      "backbone.embedding.stem.conv.weight", "seg_head.weight",
                      "frozen_head.weight"):
                T("      %-44s -> %s" % (k, k[7:]))
            T("    missing    by prefix: %s  (num_batches_tracked among them: %d)" % (
                dict(collections.Counter(top(k) for k in c["missing"])),
                sum(k.endswith("num_batches_tracked") for k in c["missing"])))
            T("    unexpected by prefix: %s" % dict(collections.Counter(top(k) for k in c["unexpected"])))
            filled = [k for k in msd if k.startswith("frozen_backbone.")
                      and k.endswith("num_batches_tracked") and k not in c["keys"]
                      and k not in c["missing"]]
            T("    frozen_backbone num_batches_tracked absent but NOT reported missing "
              "(BatchNorm version-compat fill): %d -> %d + %d = %d" % (
                  len(filled), len(c["missing"]), len(filled), len(c["missing"]) + len(filled)))
            a_ok = (c["strict"] is False and len(c["missing"]) == args.expect_missing)
    if args.path == "fixed":
        T("    (a) %s" % ("PASS" if a_ok else "FAIL"))
    else:
        T("    (a) %s" % (("DEFECT REPRODUCED: strict=False, %d missing" % len(hc[0]["missing"]))
                          if a_ok else "defect NOT reproduced"))
    T("")

    # ------------------------------------------------------------ (b)
    T("(b) every tensor of model.state_dict() vs the checkpoint tensor of the SAME name")
    per = collections.defaultdict(lambda: [0, 0, 0])      # equal, differ, absent
    casts = 0
    for k, v in msd.items():
        if k not in rsd:
            per[top(k)][2] += 1
            continue
        r = rsd[k]
        casts += r.dtype != v.dtype
        per[top(k)][0 if torch.equal(v, r.to(v.dtype)) else 1] += 1
    extra = [k for k in rsd if k not in msd]
    for p in ("backbone", "seg_head", "frozen_backbone", "frozen_head"):
        e, d, ab = per.get(p, [0, 0, 0])
        T("    %-16s %3d equal  %3d differ  %d absent" % (p, e, d, ab))
    n_eq = sum(v[0] for v in per.values())
    T("    total %d / %d model tensors equal; checkpoint tensors not in the model: %d; "
      "dtype casts needed: %d" % (n_eq, len(msd), len(extra), casts))
    bb = [k for k in rsd if k.startswith("backbone.")]
    ck_diff, el, el_diff = 0, 0, 0
    for k in bb:
        s, f = rsd[k], rsd["frozen_" + k]
        if not torch.equal(s.to(f.dtype), f):
            ck_diff += 1
        if s.is_floating_point():
            el += s.numel()
            el_diff += int((s.to(f.dtype) != f).sum())
    m_is_bb = sum(torch.equal(msd[k], rsd[k].to(msd[k].dtype)) for k in bb)
    m_is_fr = sum(torch.equal(msd[k].to(rsd["frozen_" + k].dtype), rsd["frozen_" + k]) for k in bb)
    T("    counter-check, inside the checkpoint: backbone.* differs from frozen_backbone.* "
      "(student rounded to fp16) in %d / %d tensors, %d / %d float elements (%.1f %%)" % (
          ck_diff, len(bb), el_diff, el, 100.0 * el_diff / max(el, 1)))
    T("    model trunk after this path: equals checkpoint backbone.* in %d / %d, "
      "equals checkpoint frozen_backbone.* in %d / %d" % (m_is_bb, len(bb), m_is_fr, len(bb)))
    head_is_ck = all(torch.equal(msd[k], rsd[k]) for k in ("seg_head.weight", "seg_head.bias"))
    head_is_rel = all(torch.equal(msd[k], rel[k]) for k in ("seg_head.weight", "seg_head.bias"))
    T("    model seg_head: equals checkpoint seg_head.* %s; equals the RELEASED head "
      "(weights/nuscenes-semseg-pt-v3m1-0-base) %s" % (head_is_ck, head_is_rel))
    b_ok = (n_eq == len(msd) == len(rsd) and not extra and m_is_bb == len(bb) and ck_diff > 0)
    if args.path == "fixed":
        T("    (b) %s" % ("PASS" if b_ok else "FAIL"))
    else:
        T("    (b) %s" % (("DEFECT REPRODUCED: the trunk is the frozen anchor in %d / %d tensors, "
                           "%d model tensors differ from the checkpoint"
                           % (m_is_fr, len(bb), len(msd) - n_eq))
                          if (m_is_fr == len(bb) and m_is_bb < len(bb)) else
                          "defect NOT reproduced"))
    T("")

    # ------------------------------------------------------------ (c)
    T("(c) optimizer / scheduler / scaler vs the checkpoint (independent load)")
    osd, rosd = trainer.optimizer.state_dict(), ref["optimizer"]
    groups_ok = R._same(osd["param_groups"], rosd["param_groups"])
    idx_ok = osd["state"].keys() == rosd["state"].keys()
    bad, steps = [], collections.Counter()
    alias = 0
    for i, s in rosd["state"].items():
        cur = osd["state"].get(i, {})
        steps[float(s["step"])] += 1
        for key in s:
            if not R._same(cur.get(key), s[key]):
                bad.append((i, key))
        if "exp_avg" in cur and cur["exp_avg"].data_ptr() == s["exp_avg"].data_ptr():
            alias += 1
    T("    param_groups equal (lr, betas, weight_decay, initial/max/min lr, param indices): %s"
      % groups_ok)
    T("    group sizes %s; state entries %d (checkpoint %d), same indices: %s" % (
        [len(g["params"]) for g in osd["param_groups"]], len(osd["state"]),
        len(rosd["state"]), idx_ok))
    T("    step values: %s;  entries with any of step/exp_avg/exp_avg_sq different: %d;  "
      "storages shared with the reference: %d" % (dict(steps), len(bad), alias))
    for i in list(rosd["state"])[:3]:
        c_, r_ = osd["state"][i], rosd["state"][i]
        T("      state[%d]: step %s/%s  |exp_avg| %.6e/%.6e  |exp_avg_sq| %.6e/%.6e  (restored/ckpt)"
          % (i, float(c_["step"]), float(r_["step"]), c_["exp_avg"].norm().item(),
             r_["exp_avg"].norm().item(), c_["exp_avg_sq"].norm().item(),
             r_["exp_avg_sq"].norm().item()))
    ssd, rssd = trainer.scheduler.state_dict(), ref["scheduler"]
    sched_ok = R._same(ssd, rssd)
    T("    scheduler state_dict equal: %s  (last_epoch %s/%s, _step_count %s/%s, total_steps %s/%s)"
      % (sched_ok, ssd.get("last_epoch"), rssd.get("last_epoch"), ssd.get("_step_count"),
         rssd.get("_step_count"), ssd.get("total_steps"), rssd.get("total_steps")))
    live_lr = [g["lr"] for g in trainer.optimizer.param_groups]
    T("    optimizer lr now %s == scheduler last lr %s: %s" % (
        live_lr, trainer.scheduler.get_last_lr(), live_lr == trainer.scheduler.get_last_lr()))
    if cfg.enable_amp:
        sc_ok = trainer.scaler.state_dict() == ref["scaler"]
        T("    scaler state_dict %s == checkpoint %s: %s" % (
            trainer.scaler.state_dict(), ref["scaler"], sc_ok))
    else:
        sc_ok = True
        T("    scaler: enable_amp False")
    c_ok = groups_ok and idx_ok and not bad and alias == 0 and sched_ok and sc_ok
    T("    (c) %s" % ("PASS" if c_ok else "FAIL"))
    T("")

    # ------------------------------------------------------------ (d)
    T("(d) epoch / best metric / what the hooks saw")
    iw = [h for h in trainer.hooks if isinstance(h, InformationWriter)]
    iw_iter = iw[0].curr_iter if iw else None
    d_ok = (trainer.start_epoch == ref["epoch"]
            and trainer.best_metric_value == ref["best_metric_value"]
            and iw_iter == trainer.start_epoch * len(trainer.train_loader))
    T("    start_epoch %r (checkpoint %r), max_epoch %r -> epochs left: %s" % (
        trainer.start_epoch, ref["epoch"], trainer.max_epoch,
        list(range(trainer.start_epoch + 1, trainer.max_epoch + 1)) or "none"))
    T("    best_metric_value %r (checkpoint %r)" % (trainer.best_metric_value,
                                                    ref["best_metric_value"]))
    T("    InformationWriter iteration counter %r == start_epoch * len(train_loader) = %d * %d: %s"
      % (iw_iter, trainer.start_epoch, len(trainer.train_loader),
         iw_iter == trainer.start_epoch * len(trainer.train_loader)))
    km = R.unwrap(trainer.model)
    T("    kl_enabled %s, kl_lambda %r%s" % (
        km.kl_enabled, km.kl_lambda, ("  <- %s" % rep["kl_lambda"]) if rep else ""))
    if km.kl_enabled and args.path == "fixed":
        aud = os.path.join(run_dir, "rare_class_audit.jsonl")
        want = [json.loads(ln)["kl_lambda"] for ln in open(aud) if ln.strip()
                and json.loads(ln)["epoch"] <= ref["epoch"]]
        kl_ok = bool(want) and len(set(want)) == 1 and km.kl_lambda == want[-1]
        T("    kl_lambda vs %s (epochs <= %d, read here independently): %r -> %s" % (
            aud, ref["epoch"], sorted(set(want)), "equal" if kl_ok else "NOT equal"))
        d_ok = d_ok and kl_ok
    T("    (d) %s" % ("PASS" if d_ok else "FAIL"))
    T("")

    # ------------------------------------------------------------ verdict
    if args.path == "fixed":
        ok = a_ok and b_ok and c_ok and d_ok
        T("VERDICT: %s -- resume via tools/train_distil_v2_resume.py restores this checkpoint "
          "exactly" % ("PASS" if ok else "FAIL"))
    else:
        ok = a_ok and m_is_fr == len(bb) and m_is_bb < len(bb) and head_is_rel
        T("VERDICT: %s -- Pointcept's resume restores the bookkeeping (c: %s, d: %s) but the "
          "student it resumes is the released model: trunk = frozen anchor %d/%d, head = "
          "released head %s, %d missing keys" % (
              "NEGATIVE CONTROL REPRODUCED" if ok else "NEGATIVE CONTROL NOT REPRODUCED",
              "pass" if c_ok else "fail", "pass" if d_ok else "fail", m_is_fr, len(bb),
              head_is_rel, len(hc[0]["missing"]) if hc else -1))
    if args.stated_epoch is not None and (args.stated_epoch != ref["epoch"]
                                          or args.stated_best != ref["best_metric_value"]):
        T("NOTE: this file is not the epoch-%r checkpoint the request describes: it holds epoch %r"
          " / best %r.  (d) is therefore checked against the file's own values." % (
              args.stated_epoch, ref["epoch"], ref["best_metric_value"]))
    T("wall %.1f s, peak RSS %.2f GB" % (time.time() - t0,
                                          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    trainer.writer.close()
    T.save()
    return 0 if ok else 1


KL_RUN = "/data/livo_sem/exp/sk2/armB0_s1"     # a KL arm with a calibrated lambda


def run_guards(T, args, ckpt, run_dir, save_path):
    """Every refusal path of the resume tool, driven with the real config and checkpoint."""
    res = []

    def check(name, ok, detail):
        ok = bool(ok)
        res.append(ok)
        T("%s %-4s %s" % (name, "PASS" if ok else "FAIL", detail))

    def raises(fn, *a, **k):
        try:
            fn(*a, **k)
        except (RuntimeError, ValueError) as e:
            return str(e)
        return None

    T("--path guards: the resume tool's refusals, on the real config and checkpoint")
    T("")
    # G1 -- the CLI refuses the options that re-enable Pointcept's loader
    for extra, want in ((["resume=True"], "replaces resume"), (["weight=%s" % ckpt], "replaces weight"),
                        (None, "required: --resume-from")):
        argv = ["train_distil_v2_resume.py", "--config-file", args.config]
        if extra is not None:
            argv += ["--resume-from", ckpt, "--options", "save_path=/nonexistent"] + extra
        else:
            argv += ["--options", "save_path=/nonexistent"]
        buf, code = io.StringIO(), None
        old_argv = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stderr(buf):
                R.main()
        except SystemExit as e:
            code = e.code
        finally:
            sys.argv = old_argv
        msg = buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else ""
        check("G1", code == 2 and want in msg,
              "%-40s -> exit %s: %s" % (" ".join(argv[5:]) if extra else "(no --resume-from)",
                                         code, msg))

    # G2 -- config.py is not re-dumped; a pre-existing train.log is appended to, not truncated
    marker = "MARKER written before the resume -- must survive"
    with open(os.path.join(save_path, "train.log"), "w") as fh:
        fh.write(marker + "\n")
    cfg = default_config_parser(args.config, {"save_path": save_path, "resume": True})
    check("G2", not os.path.exists(os.path.join(save_path, "config.py")),
          "parsing with the tool's options wrote no config.py into save_path")
    trainer, rep = R.build_resumed_trainer(cfg, ckpt, trainer_builder=lambda c: cpu_trainer_class(c)(c),
                                           run_dir=run_dir, require_same_save_path=False)
    txt = open(os.path.join(save_path, "train.log")).read()
    check("G2", txt.startswith(marker) and "=> RESUME model: strict=True" in txt,
          "train.log still starts with the marker and now holds the RESUME lines (%d bytes)" % len(txt))

    # G3 -- the resumed run must point at its own run dir
    e = raises(R.build_resumed_trainer, default_config_parser(
        args.config, {"save_path": save_path, "resume": True}), ckpt)
    check("G3", e is not None and "is not the run" in e, "save_path != checkpoint's run dir -> %s" % e)

    # G4 -- a config that differs from the run's config.py is refused (control first)
    c2 = default_config_parser(args.config, {"save_path": save_path, "resume": True})
    c2.resume, c2.weight = False, None
    e0 = raises(R.check_config_matches_run, c2, run_dir)
    c2.optimizer.lr = 0.0003
    e1 = raises(R.check_config_matches_run, c2, run_dir)
    diff = [ln for ln in (e1 or "").splitlines() if ln[:1] in "+-" and "lr=" in ln]
    check("G4", e0 is None and e1 is not None and diff,
          "unchanged config accepted; optimizer.lr 0.0002 -> 0.0003 refused, diff %s" % diff)

    # G5 -- restore refuses a trainer whose CheckpointLoader would still load, or that started
    trainer.cfg.weight = ckpt
    e = raises(R.restore_training_state, trainer, ckpt, run_dir)
    trainer.cfg.weight = None
    check("G5", e is not None and "CheckpointLoader" in e, "cfg.weight set -> %s" % (e or "")[:90])
    trainer.storage = object()
    e = raises(R.restore_training_state, trainer, ckpt, run_dir)
    del trainer.storage
    check("G5", e is not None and "started" in e, "EventStorage exists -> %s" % e)

    ref = torch.load(ckpt, map_location="cpu", weights_only=False)
    rsd = ref["state_dict"]
    model = R.unwrap(trainer.model)

    # G6 -- a genuine all-keys "module." prefix is stripped; a mix is refused
    pref = collections.OrderedDict(("module." + k, v) for k, v in rsd.items())
    r6 = R.restore_model(trainer.model, pref)
    check("G6", r6["prefix_stripped"] == len(rsd) and r6["missing"] == r6["unexpected"] == 0,
          "all %d keys 'module.'-prefixed -> stripped %d, strict load 0/0, every tensor equal"
          % (len(rsd), r6["prefix_stripped"]))
    mixed = collections.OrderedDict(rsd)
    k0 = next(iter(mixed))
    mixed["module." + k0] = mixed.pop(k0)
    e = raises(R.restore_model, trainer.model, mixed)
    check("G6", e is not None and "mixes" in e, "1 prefixed key among %d -> %s" % (len(rsd), e))

    # G7 -- the optimizer / scheduler structural checks
    so = copy.deepcopy(ref["optimizer"])
    i0 = next(iter(so["state"]))
    so["state"][i0]["exp_avg"] = so["state"][i0]["exp_avg"].reshape(-1)[:1].clone()
    e = raises(R.check_optimizer_compat, trainer.optimizer, so)
    check("G7", e is not None and "exp_avg" in e, "one moment with a wrong shape -> %s" % (e or "")[:110])
    flat = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    frozen_idx = next(i for i, p in enumerate(flat) if not p.requires_grad)
    so = copy.deepcopy(ref["optimizer"])
    so["state"][frozen_idx] = {"step": torch.tensor(1.0),
                               "exp_avg": torch.zeros_like(flat[frozen_idx], dtype=torch.float32),
                               "exp_avg_sq": torch.zeros_like(flat[frozen_idx], dtype=torch.float32)}
    e = raises(R.check_optimizer_compat, trainer.optimizer, so)
    check("G7", e is not None and "frozen" in e, "moments for frozen param #%d -> %s" % (frozen_idx, (e or "")[:90]))
    ss = dict(ref["scheduler"])
    ss["total_steps"] += 1
    e = raises(R.check_scheduler_compat, trainer.scheduler, ss)
    check("G7", e is not None and "total_steps" in e, "total_steps +1 -> %s" % (e or "")[:100])

    # G8 -- kl_lambda of a KL arm: restored exactly from the audit, or the resume is refused
    kl0 = (model.kl_enabled, model.kl_lambda)
    model.kl_enabled, model.kl_lambda = True, None
    msg = R.restore_kl_lambda(model, KL_RUN, 10)
    want = [json.loads(ln)["kl_lambda"] for ln in open(os.path.join(KL_RUN, "rare_class_audit.jsonl"))
            if ln.strip()]
    check("G8", model.kl_lambda == want[-1], "calibrated arm -> %s" % msg)
    model.kl_lambda = None
    e = raises(R.restore_kl_lambda, model, save_path, 10)
    check("G8", e is not None and model.kl_lambda is None, "no audit -> %s" % (e or "")[:120])
    model.kl_enabled, model.kl_lambda = kl0

    # G9 -- the T1 key set itself (every key cut by 7 characters) cannot get through strict=True
    sliced = collections.OrderedDict((k[7:], v) for k, v in rsd.items())
    e = raises(R.restore_model, trainer.model, sliced)
    check("G9", e is not None and "Missing key" in e and "Unexpected key" in e,
          "[k[7:] for k in checkpoint] -> RuntimeError (%s ...).  NB torch copies the matching "
          "keys BEFORE raising, so the exception must never be caught and ignored -- the tool "
          "lets it end the process" % (e or "").splitlines()[0][:60])

    # G10 -- the queue's exact command line reaches main_worker intact; launch() is stubbed, so
    #        nothing is built or trained, and the run's config.py must not be rewritten
    cfg_py = os.path.join(run_dir, "config.py")
    before = (os.stat(cfg_py).st_mtime_ns, sha256(cfg_py))
    argv = ["train_distil_v2_resume.py", "--config-file", args.config, "--resume-from", ckpt,
            "--options", "save_path=%s" % run_dir]
    captured = {}
    real_launch, old_argv = R.launch, sys.argv
    R.launch = lambda main_func, **kw: captured.update(kw, main_func=main_func)
    sys.argv = argv
    try:
        R.main()
    finally:
        R.launch, sys.argv = real_launch, old_argv
    after = (os.stat(cfg_py).st_mtime_ns, sha256(cfg_py))
    qc, qr = captured["cfg"]
    ok10 = (captured["main_func"] is R.main_worker and qr == ckpt and qc.resume is True
            and qc.weight is None and os.path.realpath(qc.save_path) == os.path.realpath(run_dir)
            and before == after and captured["num_gpus_per_machine"] == 1)
    qc.resume, qc.weight = False, None
    e = raises(R.check_config_matches_run, qc, run_dir)
    check("G10", ok10 and e is None,
          "queue argv -> launch(main_worker, cfg=(cfg, %s)); parse-time resume=%s weight=%s; "
          "run config.py untouched (mtime+sha256): %s; config check on it: %s"
          % (os.path.basename(qr), True, None, before == after, "accepted" if e is None else e))

    T("")
    T("GUARDS: %d / %d PASS" % (sum(res), len(res)))
    trainer.writer.close()
    return all(res)


if __name__ == "__main__":
    sys.exit(main())
