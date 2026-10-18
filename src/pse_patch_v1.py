"""
PSE patch v1: TA uses CANN npu_fusion_attention(pse) flash kernel.
Replaces F.sdpa(float mask) which triggers CANN fallback (2-10x penalty).
APB (diffusion module) unchanged for now (keeps naive).

Scale: Protenix _attention uses softmax(QK^T + bias) with no 1/sqrt(d).
CANN with scale=1.0: softmax(1.0*(QK^T + pse)) = same thing. No pre-scaling.
"""
import os
import torch
import torch.nn.functional as F

if os.environ.get("SDPA_DISABLE", "0") == "1":
    print("[SDPA] disabled")
else:
    import torch_npu

    _DEBUG = os.environ.get("SDPA_DEBUG", "0") == "1"
    _COMPARE_N = int(os.environ.get("PSE_COMPARE_N", "0"))
    _COMPARE_RETURN = os.environ.get("PSE_COMPARE_RETURN", "split")
    _SPLIT_MODE = os.environ.get(
        "PSE_SPLIT_MODE", "pse_native"
    )
    _n_pse = 0
    _n_skip = 0
    _n_mask_fallback = 0

    def _attention_pse(query, key, value, biases):
        global _n_pse, _n_skip
        d = query.shape[-1]

        # APB path (d>=40): keep naive (precision-sensitive diffusion module)
        if d >= 40:
            _n_skip += 1
            key_t = key.transpose(-2, -1)
            a = torch.matmul(query, key_t)
            for b in biases:
                a = a + b
            with torch.amp.autocast("cuda", enabled=False):
                a = F.softmax(a, dim=-1)
            return torch.matmul(a, value)

        # TA path (d=32): flash attention via npu_fusion_attention(pse)
        _n_pse += 1
        H = query.shape[1]

        if _DEBUG and _n_pse <= 5:
            for i, b in enumerate(biases):
                print(
                    f"[PSE] call {_n_pse} bias[{i}]: shape={tuple(b.shape)} "
                    f"stride={tuple(b.stride())} dtype={b.dtype} "
                    f"contiguous={b.is_contiguous()} numel={b.numel()}",
                    flush=True,
                )
                if b.numel():
                    print(
                        f"[PSE] call {_n_pse} bias[{i}] stats: "
                        f"min={b.min().item():.6e} max={b.max().item():.6e} "
                        f"mean={b.float().mean().item():.6e} "
                        f"finite={torch.isfinite(b).all().item()}",
                        flush=True,
                    )
            if len(biases) >= 2:
                b0, b1 = biases[0], biases[1]
                b0_binary = ((b0 == 0) | (b0 == -1.0e9)).all().item()
                all_masked = (b0 < -1.0e8).all(dim=-1)
                print(
                    f"[PSE] call {_n_pse} real-mask: binary={b0_binary} "
                    f"all_masked_rows={all_masked.float().mean().item():.6f} "
                    f"keep_ratio={(b0 >= -1.0e8).float().mean().item():.6f}",
                    flush=True,
                )

        # Protenix TriangleAttention receives two broadcastable biases:
        #   row mask [B,1,1,S] and pair bias [1,H,S,S].  Materializing their
        # sum creates an [B,H,S,S] tensor (about 11GB for the real shape).
        # CANN accepts the two roles separately: bool atten_mask plus an
        # expanded pse view, avoiding that full-size intermediate.
        pse_parts = []
        atten_mask = None
        for bias in biases:
            is_row_mask = (
                bias.ndim == 4
                and bias.shape[1] == 1
                and bias.shape[2] == 1
                and bias.shape[3] == key.shape[-2]
            )
            if is_row_mask:
                if bias.dtype == torch.bool:
                    atten_mask = bias
                else:
                    # The Protenix mask uses 0 for keep and -1e9 for masked.
                    atten_mask = bias >= -1.0e8
            else:
                pse_parts.append(bias)

        if pse_parts:
            total_bias = pse_parts[0]
            for b in pse_parts[1:]:
                total_bias = total_bias + b
            if total_bias.dtype != query.dtype:
                total_bias = total_bias.to(query.dtype)
        else:
            total_bias = None

        bias_shape = (*query.shape[:-1], key.shape[-2])
        if total_bias is not None:
            if _SPLIT_MODE in (
                "both_expanded", "pse_expanded_mask_native", "pse_only",
            ):
                total_bias = total_bias.expand(bias_shape)
        if atten_mask is not None:
            if _SPLIT_MODE == "pse_native":
                # The CANN broadcast forms for atten_mask either reject the
                # Protenix shape or differ numerically.  Dropping the mask is
                # exact only when every key is visible, so guard that here.
                if not atten_mask.all().item():
                    _n_mask_fallback += 1
                    return _orig(query, key, value, biases)
                atten_mask = None
            if _SPLIT_MODE == "pse_only":
                atten_mask = None
            if _SPLIT_MODE in ("both_expanded", "mask_expanded_pse_native"):
                atten_mask = atten_mask.expand(bias_shape)
        if _SPLIT_MODE not in (
            "both_expanded",
            "pse_expanded_mask_native",
            "pse_only",
            "pse_native",
            "mask_expanded_pse_native",
        ):
            raise ValueError(f"Invalid PSE_SPLIT_MODE: {_SPLIT_MODE}")

        if _DEBUG and _n_pse <= 5:
            tb = f"shape={total_bias.shape}" if total_bias is not None else "None"
            print(f"[PSE] call {_n_pse}: q={query.shape} d={d} H={H} bias={tb}")

        full_reference = None
        if _n_pse <= _COMPARE_N:
            full_bias = biases[0]
            for b in biases[1:]:
                full_bias = full_bias + b
            if full_bias.dtype != query.dtype:
                full_bias = full_bias.to(query.dtype)
            full_reference = torch_npu.npu_fusion_attention(
                query, key, value,
                H, "BNSD",
                pse=full_bias,
                scale=1.0,
            )[0]

        # CANN flash attention with additive bias (pse)
        # scale=1.0 because Protenix doesn't use 1/sqrt(d) in _attention
        out = torch_npu.npu_fusion_attention(
            query, key, value,
            H, "BNSD",
            pse=total_bias,
            atten_mask=atten_mask,
            scale=1.0,
        )[0]

        if full_reference is not None:
            diff = (out - full_reference).abs()
            print(
                f"[PSE-COMPARE] call={_n_pse} "
                f"split_mode={_SPLIT_MODE} "
                f"max_abs_diff={diff.max().item():.6e} "
                f"mean_abs_diff={diff.mean().item():.6e} "
                f"out_min={out.min().item():.6e} "
                f"out_max={out.max().item():.6e} "
                f"ref_min={full_reference.min().item():.6e} "
                f"ref_max={full_reference.max().item():.6e}",
                flush=True,
            )
            if _COMPARE_RETURN == "full":
                return full_reference

        return out

    import protenix.model.triangular.layers as _layers
    _orig = _layers._attention
    _layers._attention = _attention_pse
    _layers._attention_original = _orig

    import atexit
    def _report():
        print(
            f"[PSE] TA->flash(pse): {_n_pse} calls | "
            f"mask-fallback: {_n_mask_fallback} | APB->naive: {_n_skip} calls"
        )
    atexit.register(_report)
    print("[PSE] TA->npu_fusion_attention(pse+bool mask), APB->naive")
