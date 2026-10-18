#!/usr/bin/env python3
"""Protenix entry point for the FlagOS Ascend container.

Import RDKit before torch_npu, keep the proven CANN fusion-attention path,
and make FlagGems opt-in so the first FlagOS retest is comparable with the host
36.14s result.
"""
import os
import sys

import rdkit.Chem  # noqa: F401  # must precede torch_npu

import npu_shim  # noqa: E402,F401

if os.environ.get("FLAGOS_ENABLE_FLAGGEMS", "0") == "1":
    import flag_gems

    ops_env = os.environ.get("FLAGGEMS_OPS", "").strip()
    if ops_env:
        ops = tuple(x.strip() for x in ops_env.split(",") if x.strip())
        try:
            flag_gems.only_enable(ops)
        except (TypeError, AttributeError):
            try:
                flag_gems.enable(only=ops)
            except TypeError:
                flag_gems.enable()
        print(f"[FlagOS] FlagGems selective kernels enabled: {ops}", flush=True)
    else:
        flag_gems.enable()
        print(f"[FlagOS] FlagGems enabled: {getattr(flag_gems, '__version__', '?')}", flush=True)
else:
    try:
        import flag_gems
        print(
            f"[FlagOS] FlagGems present but not overriding kernels: "
            f"{getattr(flag_gems, '__version__', '?')}",
            flush=True,
        )
    except Exception as exc:
        print(f"[FlagOS] FlagGems unavailable: {exc}", flush=True)

import apb_pse_patch_v2  # noqa: E402,F401
import pse_patch_v1  # noqa: E402,F401  # TriangleAttention PSE (SDPA_DISABLE=1 to disable)
import fg_triatt_patch  # noqa: E402,F401  # optional streaming TA backend
import ta_projection_patch  # noqa: E402,F401  # fuse TA input projections
import tm_npu_patch  # noqa: E402,F401  # TriangleMultiplicativeUpdate fused linear
import triatt_npu_patch  # noqa: E402,F401  # NPU-compatible Triton TA (used with --triatt_kernel triattention)
import amp_mixed_patch  # noqa: E402,F401
import static_cache_patch  # noqa: E402,F401
import metax_pair_prof  # noqa: E402,F401  # pairformer breakdown (PAIR_PROF_ENABLE=1)

# Some Protenix checkouts still import the v1 module name in the CLI entry
# point. Aliasing it prevents that import from replacing the proven v2 patch.
sys.modules.setdefault("apb_pse_patch", apb_pse_patch_v2)
# runner.batch_inference may import sdpa_patch later; keep TriangleAttention on
# the instrumented/proven pse_patch_v1 implementation instead of the older copy.
sys.modules.setdefault("sdpa_patch", pse_patch_v1)

from runner.batch_inference import protenix_cli  # noqa: E402


if __name__ == "__main__":
    protenix_cli()
