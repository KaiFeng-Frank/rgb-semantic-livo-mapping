"""
VERIFIED PTv3 loader — measured, not guessed.

Established facts (each one tested on this machine, 2026-09-21):
  * Pointcept HEAD (1342eda) CANNOT load the released nuScenes checkpoint:
        TypeError: PointTransformerV3.__init__() got an unexpected keyword argument 'cls_mode'
    HEAD also pulls in PointROPE, i.e. the module structure genuinely changed. Upstream says
    "Released model weights are temporarily invalid as the model structure of PTv3 is adjusted."
  * Pointcept v1.5.1 (72a7993) loads it PERFECTLY:
        488 model tensors / 488 checkpoint tensors, 0 missing, 0 unexpected, 0 shape mismatch,
        load_state_dict(strict=True) succeeds, 46.2M parameters.
  * The checkpoint is DDP-saved: every key is prefixed 'module.' — strip it.
  * pointcept.models.__init__ eagerly imports EVERY model, dragging in CUDA extensions PTv3 does
    not need (pointops, pointgroup_ops) plus ocnn. Stub the two ops modules; do NOT build them.
    The stub must raise AttributeError on dunder lookups or `inspect` breaks.

NEVER use strict=False here. If keys stop matching, the version pairing is wrong — fix that instead.
"""
import sys, types, warnings

POINTCEPT_V151 = "/data/livo_sem/src/Pointcept_v151"
WEIGHT_DIR     = "/data/livo_sem/weights/nuscenes-semseg-pt-v3m1-0-base"
CONFIG_PATH    = WEIGHT_DIR + "/config.py"
CKPT_PATH      = WEIGHT_DIR + "/model/model_best.pth"

# The 16 nuScenes classes, in checkpoint order. Index == predicted class id.
NUSCENES_CLASSES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface",
    "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]


class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self


class _Stub(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)          # keep inspect/importlib working
        return _Any


def _install_stubs():
    for n in ("pointops", "pointgroup_ops"):
        if n not in sys.modules:
            m = _Stub(n); m.__file__ = f"<stub {n}>"; sys.modules[n] = m


def build_ptv3(device="cuda", enable_flash=None):
    """Build PTv3 from the frozen HF config and load the released weights with strict=True."""
    import torch
    warnings.filterwarnings("ignore")
    if POINTCEPT_V151 not in sys.path:
        sys.path.insert(0, POINTCEPT_V151)
    _install_stubs()

    from pointcept.utils.config import Config
    from pointcept.models import build_model

    cfg = Config.fromfile(CONFIG_PATH)
    if enable_flash is not None:
        cfg.model.backbone.enable_flash = bool(enable_flash)
    model = build_model(cfg.model)

    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)      # MUST stay strict
    return model.to(device).eval(), cfg


if __name__ == "__main__":
    import torch
    m, cfg = build_ptv3(device="cuda")
    n = sum(p.numel() for p in m.parameters())
    print(f"OK  {type(m).__name__}  params={n/1e6:.1f}M  "
          f"flash={cfg.model.backbone.enable_flash}  classes={cfg.model.num_classes}")
