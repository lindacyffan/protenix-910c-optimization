"""
APB PSE patch v2: AttentionPairBias + local attention via npu_fusion_attention.

Patches primitives._attention() -- covers both standard APB path (4D) and
local/AtomTransformer attention path (3D chunked tensors from chunk_layer).

v1: 4D path uses flash kernel, 3D falls back to naive (F.sdpa fallback).
v2: 3D path reshapes to [B,1,S,d] and uses flash kernel with head_num=1.
    This covers the 6061 naive calls from AtomTransformer local attention.

Import AFTER npu_shim but BEFORE any protenix model code.

Env knobs:
    APB_PSE_DISABLE=1   skip patching
    APB_PSE_DEBUG=1     print first 5 call shapes per path
    APB_PSE_FP32=1      force fp32 for q/k/v/pse (default: keep input dtype)
    APB_PSE_BROADCAST=copy|view|keep
                        copy (default): expand + contiguous [B,H,S,S]
                        view: strided expand, no materialized copy
                        keep: pass [B,1,S,S] directly to CANN
"""
import os
import torch
import torch.nn.functional as F

_DISABLE = os.environ.get("APB_PSE_DISABLE", "0") == "1"
_DEBUG = os.environ.get("APB_PSE_DEBUG", "0") == "1"
_FORCE_FP32 = os.environ.get("APB_PSE_FP32", "0") == "1"
_BIAS_BROADCAST = os.environ.get("APB_PSE_BROADCAST", "copy").strip().lower()
_n_4d = 0
_n_3d = 0
_n_skip = 0


def _attention_apb_pse(
    q,
    k,
    v,
    attn_bias=None,
    use_efficient_implementation=True,
    inplace_safe=False,
):
    """Drop-in replacement for primitives._attention() using CANN flash kernel."""
    global _n_4d, _n_3d, _n_skip

    import torch_npu

    # ---- 3D path: local attention chunks from chunk_layer ----
    # Shape: [B, S, d] where B folds (batch, heads, n_trunks).
    # Reshape to [B, 1, S, d] and use head_num=1.
    if q.ndim == 3:
        _n_3d += 1
        d = q.shape[-1]
        B = q.shape[0]
        orig_shape = q.shape

        # Reshape qkv to 4D [B, 1, S, d]
        q4 = q.unsqueeze(-3)
        k4 = k.unsqueeze(-3)
        v4 = v.unsqueeze(-3)

        # Handle bias: [B, S, S] -> [B, 1, S, S]
        if attn_bias is not None:
            if attn_bias.ndim == 3:
                b4 = attn_bias.unsqueeze(-3)
            elif attn_bias.ndim == 2:
                b4 = attn_bias.unsqueeze(0).unsqueeze(0)
            else:
                b4 = attn_bias
            if _FORCE_FP32:
                q4 = q4.to(torch.float32)
                k4 = k4.to(torch.float32)
                v4 = v4.to(torch.float32)
                b4 = b4.to(torch.float32)
            elif b4.dtype != q4.dtype:
                b4 = b4.to(q4.dtype)
        else:
            b4 = None

        if _DEBUG and _n_3d <= 5:
            print(f"[APB-PSE 3D] call {_n_3d}: q={list(q.shape)} -> {list(q4.shape)} "
                  f"bias={'y' if b4 is not None else 'n'} dtype={q4.dtype}")

        out = torch_npu.npu_fusion_attention(
            q4, k4, v4, 1, "BNSD",
            pse=b4,
            scale=1.0,
        )[0]

        # Remove the head dim: [B, 1, S, d] -> [B, S, d]
        return out.reshape(orig_shape)

    # ---- No bias: rare, fall through to SDPA ----
    if attn_bias is None:
        _n_skip += 1
        q32 = q.to(torch.float32)
        k32 = k.to(torch.float32)
        return F.scaled_dot_product_attention(q32, k32, v, attn_mask=None, scale=1.0)

    # ---- 4D+ path: standard APB attention ----
    _n_4d += 1
    d = q.shape[-1]
    H = q.shape[-3]

    orig_shape = q.shape
    need_reshape = q.ndim > 4
    if need_reshape:
        q = q.reshape(-1, H, q.shape[-2], d)
        k = k.reshape(-1, H, k.shape[-2], d)
        v = v.reshape(-1, H, v.shape[-2], d)

    # Handle attn_bias shape: ensure [B, H, Q, K].
    if attn_bias.ndim > 4:
        attn_bias = attn_bias.reshape(
            -1, attn_bias.shape[-3], attn_bias.shape[-2], attn_bias.shape[-1]
        )
    elif attn_bias.ndim == 3:
        attn_bias = attn_bias.unsqueeze(-3)

    if attn_bias.shape[-3] == 1 and H != 1:
        if _BIAS_BROADCAST == "keep":
            # Pass [B, 1, S, S] directly; let CANN broadcast if supported.
            pass
        elif _BIAS_BROADCAST == "view":
            # Strided view without materializing the full [B, H, S, S] tensor.
            attn_bias = attn_bias.expand(-1, H, -1, -1)
        else:
            attn_bias = attn_bias.expand(-1, H, -1, -1).contiguous()

    if _FORCE_FP32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        attn_bias = attn_bias.to(torch.float32)
    else:
        if attn_bias.dtype != q.dtype:
            attn_bias = attn_bias.to(q.dtype)

    if _DEBUG and _n_4d <= 5:
        print(f"[APB-PSE 4D] call {_n_4d}: q={list(q.shape)} d={d} H={H} "
              f"bias={list(attn_bias.shape)} dtype={q.dtype} "
              f"broadcast={_BIAS_BROADCAST} contiguous={attn_bias.is_contiguous()}")

    out = torch_npu.npu_fusion_attention(
        q, k, v, H, "BNSD",
        pse=attn_bias,
        scale=1.0,
    )[0]

    if need_reshape:
        out = out.reshape(orig_shape)
    return out


# Apply the patch
import protenix.model.modules.primitives as _prim

_orig = _prim._attention
if not _DISABLE:
    _prim._attention = _attention_apb_pse
    _prim._attention_original = _orig

    import atexit

    def _report():
        print(f"[APB-PSE v2] 4D flash: {_n_4d} | 3D flash: {_n_3d} | naive: {_n_skip}")
    atexit.register(_report)
    print("[APB-PSE v2] primitives._attention -> npu_fusion_attention(pse) [4D+3D]")
else:
    print("[APB-PSE v2] disabled by APB_PSE_DISABLE=1")
