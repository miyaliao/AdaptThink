#!/bin/bash
# Reproduce the environment-side changes needed to run this fork's RL loop here.
#
# None of this is algorithm code: it is the set of adaptations that make an
# 8-GPU-oriented verl fork run on this cluster (H200, vLLM 0.8.5, torch 2.6,
# CUDA_VISIBLE_DEVICES holding GPU UUIDs, no flash-attn build). They live outside
# the repository by nature (site-packages, conda env), so they are captured here as
# a single idempotent script.
#
# Run once per environment:
#     bash scripts/setup_env_shims.sh
#
# Requires PYTHON to point at the training environment, e.g.
#     PYTHON=/home/n2602358e/.conda/envs/nt/bin/python bash scripts/setup_env_shims.sh
#
# What it does:
#   1. src/config/__init__.py        - hydra needs the config directory to be a package
#   2. verl attn_implementation      - 'flash_attention_2' -> 'sdpa' (no flash-attn build)
#   3. verl torch_functional         - disable the flash cross-entropy path (shape mismatch)
#   4. sitecustomize.py              - vLLM cannot parse GPU UUIDs in CUDA_VISIBLE_DEVICES
#   5. flash_attn API shim           - import-compatible stand-in, kernels never executed
#   6. lab_rl_1gpu.pbs               - single-GPU variant of the 2-GPU RL job

set -euo pipefail

PYTHON="${PYTHON:-python3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"

echo "repo : $REPO"
echo "python: $PYTHON"
echo "site : $SITE"

# ---------------------------------------------------------------- 1
touch "$REPO/src/config/__init__.py"
echo "[1/6] src/config/__init__.py"

# ---------------------------------------------------------------- 2
"$PYTHON" - "$REPO" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
patched = []
for sub in ("verl", "src"):
    for path in (root / sub).rglob("*.py"):
        text = path.read_text()
        new = text.replace("attn_implementation='flash_attention_2'", "attn_implementation='sdpa'")
        new = new.replace('attn_implementation="flash_attention_2"', "attn_implementation='sdpa'")
        if new != text:
            path.write_text(new)
            patched.append(str(path.relative_to(root)))
print(f"[2/6] attn_implementation -> sdpa in {len(patched)} file(s): {', '.join(patched) or 'already done'}")
PY

# ---------------------------------------------------------------- 3
"$PYTHON" - "$REPO" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "verl/utils/torch_functional.py"
text = path.read_text()
old = "    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True"
new = ("    # The local flash-attn is an API shim without the Triton kernel, so its\n"
       "    # cross_entropy_loss does not return per-token losses as verl expects here.\n"
       "    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False")
if old in text:
    path.write_text(text.replace(old, new))
    print("[3/6] flash cross-entropy path disabled")
else:
    print("[3/6] already patched")
PY

# ---------------------------------------------------------------- 4
cat > "$SITE/sitecustomize.py" <<'PYEOF'
"""Patch vLLM's device-id parsing when CUDA_VISIBLE_DEVICES holds GPU UUIDs.

This cluster exports CUDA_VISIBLE_DEVICES as "GPU-9ce00630-...", and vLLM 0.8.5
casts those entries to int, which crashes inside ray worker processes as soon as
verl imports vllm.model_executor.layers.fused_moe. Python imports sitecustomize in
every interpreter, so this also reaches ray's workers, which a normal launcher
cannot.
"""
import os


def _patch_uuid_devices():
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
        return
    try:
        import vllm.platforms.cuda as cuda_platform
    except Exception:
        return
    if getattr(cuda_platform, "_uuid_patched", False):
        return

    def logical_id(device_id=None):
        if device_id is None:
            return 0
        if isinstance(device_id, int):
            return device_id
        try:
            return int(device_id)
        except (TypeError, ValueError):
            return 0

    cuda_platform.device_id_to_physical_device_id = logical_id
    cuda_platform._uuid_patched = True


_patch_uuid_devices()
PYEOF
echo "[4/6] sitecustomize.py installed"

# ---------------------------------------------------------------- 5
"$PYTHON" - "$SITE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
pkg = root / "flash_attn"
for d in ("", "layers", "ops", "ops/triton", "modules"):
    (pkg / d).mkdir(parents=True, exist_ok=True)

(pkg / "_stub.py").write_text('''
def _make_stub(name):
    def _stub(*args, **kwargs):
        raise NotImplementedError(
            f"flash-attn '{name}' is not compiled in this environment; "
            "this run uses attn_implementation='sdpa'."
        )
    _stub.__name__ = name
    return _stub


def module_getattr(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return _make_stub(name)
''')

(pkg / "__init__.py").write_text('''
from flash_attn._stub import module_getattr

__version__ = "2.7.4.post1"

flash_attn_func = module_getattr("flash_attn_func")
flash_attn_varlen_func = module_getattr("flash_attn_varlen_func")
flash_attn_qkvpacked_func = module_getattr("flash_attn_qkvpacked_func")
flash_attn_kvpacked_func = module_getattr("flash_attn_kvpacked_func")
flash_attn_with_kvcache = module_getattr("flash_attn_with_kvcache")

__getattr__ = module_getattr
''')

(pkg / "bert_padding.py").write_text('''
import torch
from einops import rearrange  # noqa: F401  (re-exported, as flash-attn does)


def index_first_axis(x, indices):
    return x.reshape(-1, *x.shape[2:])[indices]


def unpad_input(hidden_states, attention_mask, use_padding_free_transformer=False):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_len = int(seqlens.max().item())
    cu = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return index_first_axis(hidden_states, indices), indices, cu, max_len


def pad_input(hidden_states, indices, batch, seqlen):
    out = torch.zeros(batch * seqlen, *hidden_states.shape[1:],
                      device=hidden_states.device, dtype=hidden_states.dtype)
    out[indices] = hidden_states
    return out.view(batch, seqlen, *hidden_states.shape[1:])
''')

(pkg / "ops/triton/cross_entropy.py").write_text('''
import torch


def cross_entropy_loss(logits, labels, label_smoothing=0.0, lse=None, reduce=True,
                       ignore_index=-100, return_z_loss=False, **kwargs):
    loss = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=ignore_index, label_smoothing=label_smoothing,
        reduction="mean" if reduce else "none")
    z = torch.zeros((), device=logits.device) if reduce else torch.zeros_like(loss)
    return loss, z
''')

attr_module = "from flash_attn._stub import module_getattr\n\n__getattr__ = module_getattr\n"
for d in ("layers", "ops", "ops/triton", "modules"):
    (pkg / d / "__init__.py").write_text(attr_module)
for f in ("layers/rotary.py", "ops/rms_norm.py", "ops/layer_norm.py",
          "ops/fused_dense.py", "ops/activations.py",
          "ops/triton/layer_norm.py", "ops/triton/rotary.py"):
    (pkg / f).write_text(attr_module)

dist = root / "flash_attn-2.7.4.post1.dist-info"
dist.mkdir(exist_ok=True)
(dist / "METADATA").write_text("Metadata-Version: 2.1\nName: flash_attn\nVersion: 2.7.4.post1\n")
print(f"[5/6] flash-attn shim installed at {pkg}")
PY

# ---------------------------------------------------------------- 6
sed 's|^#PBS -l select=1:ncpus=16:ngpus=2:mem=128gb$|#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb|; s|^NGPU="${NGPU:-2}"$|NGPU="${NGPU:-1}"|; s|^#PBS -N nt_rl_small$|#PBS -N nt_rl_1gpu|; s|^#PBS -l walltime=08:00:00$|#PBS -l walltime=03:00:00|' \
    "$REPO/scripts/lab_rl_small.pbs" > "$REPO/scripts/lab_rl_1gpu.pbs"
chmod +x "$REPO/scripts/lab_rl_1gpu.pbs"
echo "[6/6] scripts/lab_rl_1gpu.pbs generated"

echo
echo "verify:"
echo "  $PYTHON -c \"from flash_attn.bert_padding import pad_input; import vllm.platforms.cuda\""
echo "  $PYTHON -c \"from verl.utils.torch_functional import FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE as f; print(f)\""
