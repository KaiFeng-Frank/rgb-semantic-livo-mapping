#!/usr/bin/env python3
"""Prepare and verify one no-KL diagnostic arm without editing shared model code."""
import copy, hashlib, json, os, sys, subprocess, datetime
from pathlib import Path
ROOT = Path("/data/wuyou/livo_sem")
OUT = ROOT / "out/v04/nokl_20260923"
CFGDIR = ROOT / "src/Pointcept_v151/configs/semantic_kitti"
SEED = 20260923
arm = sys.argv[1]
assert arm in ("D_noKL", "Rprime_noKL")
parent = "D" if arm == "D_noKL" else "Rprime"
sys.path.insert(0, str(ROOT / "src"))
import pointcept_ext
import distil_ext
if parent == "Rprime":
    import distil_ext_rprime
import numpy as np
import torch
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.datasets.utils import point_collate_fn

def seed_all():
    import random
    random.seed(SEED * 8)
    np.random.seed(SEED * 8)
    torch.manual_seed(SEED * 8)
    torch.cuda.manual_seed_all(SEED * 8)

def digest_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def flat(value, prefix=""):
    if isinstance(value, dict):
        return {key: val for k, v in value.items() for key, val in flat(v, prefix + "." + str(k)).items()}
    if isinstance(value, (list, tuple)):
        result = {prefix + ".container": type(value).__name__}
        for i, v in enumerate(value):
            result.update(flat(v, prefix + "." + str(i)))
        return result
    return {prefix: repr(value)}

OUT.mkdir(parents=True, exist_ok=True)
own = OUT / arm
own.mkdir(exist_ok=True)
old_path = ROOT / "exp/sk" / ("arm" + parent) / "config.py"
old = Config.fromfile(str(old_path))
cfg = Config(copy.deepcopy(old._cfg_dict), filename=str(old_path))
cfg.seed = SEED
cfg.save_path = str(ROOT / "exp/sk" / ("arm" + arm))
cfg.model.kl_enabled = False
cfg.model.kl_lambda = None  # inactive with KL disabled; avoids appending to historical calibration log
if parent == "Rprime":
    selector = next(t for t in cfg.data.train.transform if t["type"] == "VoxelRandomSupervise")
    selector["audit_path"] = str(own / "voxel_audit.jsonl")
a, b = flat(old._cfg_dict), flat(cfg._cfg_dict)
changes = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
allowed = {".seed", ".save_path", ".model.kl_enabled", ".model.kl_lambda",
           ".data.train.transform.5.audit_path"}
assert set(changes) <= allowed, changes
assert cfg.epoch == cfg.eval_epoch == 10 and cfg.batch_size == 2
assert cfg.num_worker == 8 and cfg.weight is None and cfg.resume is False
assert cfg.data.train.label_source == "gt"
assert cfg.data.val.split == "val" and cfg.data.test.split == "test"
assert cfg.model.freeze_backbone is False and cfg.model.exclude_classes == ()
config_path = CFGDIR / ("arm_" + arm + ".py")
assert not Path(cfg.save_path).exists(), "Refuse to overwrite an experiment"
if config_path.exists():
    assert flat(Config.fromfile(str(config_path))._cfg_dict) == flat(cfg._cfg_dict), "Prepared config changed"
else:
    cfg.dump(str(config_path))
gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                     check=True, capture_output=True, text=True)
assert not gpu.stdout.strip(), "GPU is occupied: " + gpu.stdout
seed_all()
dataset_cfg = copy.deepcopy(cfg.data.train)
if parent == "Rprime":
    next(t for t in dataset_cfg.transform if t["type"] == "VoxelRandomSupervise")["audit_path"] = str(own / "preflight_voxel_audit.jsonl")
ds = build_dataset(dataset_cfg)
ids = ["/".join(ds._seq_frame(p)) for p in ds.data_list]
assert len(ids) == 17228, len(ids)
seqs = sorted(set(s.split("/")[0] for s in ids))
assert seqs == ["00", "01", "02", "04", "05", "06", "09", "10"], seqs
selected = []
for i in (0, 1, 100, 1000):
    seed_all()
    item = ds[i]
    selected.append({"index": i, "frame": ids[i],
                     "voxels": len(item["segment"]),
                     "n_sup": int((item["segment"] >= 0).sum()),
                     "coord_sha256": hashlib.sha256(item["coord"].numpy().tobytes()).hexdigest()})
seed_all()
model = build_model(cfg.model)
h = hashlib.sha256()
for name, tensor in sorted(model.state_dict().items()):
    if name.startswith("frozen_"):
        continue
    h.update(name.encode())
    h.update(str(tensor.dtype).encode())
    h.update(str(tuple(tensor.shape)).encode())
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
initial_sha = h.hexdigest()
assert model._n_loaded == 488
model = model.cuda().train()
def forbidden_anchor(*args, **kwargs):
    raise AssertionError("No-KL arm called the frozen anchor")
model._anchor_logits16 = forbidden_anchor
seed_all()
batch = point_collate_fn([ds[0], ds[1]], mix_prob=cfg.mix_prob)
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.enable_amp):
    result = model(batch)
assert int(result["n_kl"]) == 0 and float(result["kl"]) == 0.0
assert int(result["n_sup"]) > 0
assert torch.isfinite(result["loss"])
assert torch.allclose(result["loss"].detach(), result["ce"] + result["lovasz"], atol=1e-6)
result["loss"].backward()
grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
assert any(bool((g != 0).any()) for g in grads)
assert all(p.grad is None for name, p in model.named_parameters() if name.startswith("frozen_"))
shared = ["tools/train_distil.py", "tools/cache_trained.py", "tools/arm_verdict.py",
          "src/distil_ext.py", "src/pointcept_ext.py", "src/score_2d_vs_3d.py",
          "src/Pointcept_v151/configs/semantic_kitti/semseg-pt-v3m1-distil-common9.py",
          "out/pseudo/rare_weights.json"]
normalized = copy.deepcopy(cfg._cfg_dict)
normalized["save_path"] = "<arm>"
normalized["data"]["train"]["type"] = "<supervision>"
normalized["data"]["train"]["supervise"] = "<supervision>"
ts = normalized["data"]["train"]["transform"]
ts[:] = [t for t in ts if t["type"] != "VoxelRandomSupervise"]
next(t for t in ts if t["type"] == "GridSample")["keys"] = ("coord", "strength", "segment", "frustum")
record = {
    "arm": arm, "parent": parent, "seed": SEED, "parent_seed": int(old.seed),
    "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "config": str(config_path), "config_sha256": digest_file(config_path),
    "parent_config_sha256": digest_file(old_path), "config_changes": changes,
    "runtime_env": {"NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES", "")},
    "numpy_version": np.__version__, "torch_version": torch.__version__,
    "initial_student_sha256": initial_sha, "strict_loaded_tensors": model._n_loaded,
    "train_frames": len(ds), "train_sequences": seqs,
    "training_order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    "normalized_pair_config_sha256": hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode()).hexdigest(),
    "selected_samples": selected,
    "smoke": {k: float(v.detach()) for k, v in result.items()},
    "finite_nonzero_student_gradients": True, "anchor_was_not_called": True,
    "shared_sha256": {rel: digest_file(ROOT / rel) for rel in shared},
    "protocol_sha256": digest_file(OUT / "protocol.md"),
    "pass": True,
}
if (own / "preflight.json").exists():
    with (own / "preflight_initial.json").open("x") as f:
        f.write((own / "preflight.json").read_text())
(own / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
print("PREFLIGHT_PASS", json.dumps(record, ensure_ascii=False), flush=True)
