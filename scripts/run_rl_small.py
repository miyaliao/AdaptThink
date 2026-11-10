#!/usr/bin/env python3
"""Launch verl's PPO trainer with a fix for GAAS's UUID CUDA_VISIBLE_DEVICES.

On this cluster `CUDA_VISIBLE_DEVICES` holds GPU UUIDs ("GPU-3b2bc0d3-...") instead
of indices, and vLLM 0.8.5 casts those entries to int in
`vllm.platforms.cuda.device_id_to_physical_device_id`. That crashes as soon as
verl imports `vllm.model_executor.layers.fused_moe` (the import chain reaches
`cutlass_fp8_supported()` before any of our code runs).

Patching the module attribute before importing the trainer keeps CUDA's own
mapping: with one visible UUID, logical device 0 *is* that device, so returning the
logical id is the correct physical mapping.

Invoked exactly like `python -m src.main_ppo`, i.e. all hydra overrides are passed
through on the command line.
"""

import os
import sys
from pathlib import Path


def patch_uuid_devices():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-"):
        return
    try:
        import vllm.platforms.cuda as cuda_platform
    except Exception as exc:                     # pragma: no cover
        print(f"[launcher] WARNING: could not patch vllm: {exc}", flush=True)
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
    print(f"[launcher] patched vllm uuid devices ({visible.split(',')[0][:24]}...)", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    patch_uuid_devices()
    from src.main_ppo import main
    main()
