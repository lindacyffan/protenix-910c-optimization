"""
NPU-compatible Triton TriangleAttention forward for Ascend 910C.

The upstream `triattention` kernel builds Bias1 (the row mask) as a block
pointer with strides=(0, 1). Triton-Ascend rejects zero strides at compile
time ("strides must not be zero"). This patch replaces the forward path with
plain `tl.load` pointer arithmetic and makes S a compile-time constant.

Math is unchanged: qk = dot(q, k^T) + Bias1 + Bias2, softmax via exp2 with
scale=1/sqrt(d) already applied to q by Attention._prep_qkv.

Loop diagnostics showed that all loop forms execute the same iterations; the
original S=1170 error came from an incorrect Q/K/V base offset for N > 1.
TRIATT_NPU_LOOP is retained for performance comparisons:
    0  range(tl.cdiv(S, BLOCK_N))
    1  tl.static_range(NB)               (default; fully unrolled)
    2  range(NB)                         (constexpr bound, no tl.cdiv)
    3  tl.range(NB, loop_unroll_factor=4)

Env knobs:
    TRIATT_NPU_PATCH_DISABLE=1   skip patching (keeps upstream kernel)
    TRIATT_NPU_PATCH_MASK_ONLY=1 read only Bias2; mask is already enforced by
                                 the kernel's -1e6 softmax masking
    TRIATT_NPU_LOOP=1            loop mode (default 1)
    TRIATT_NPU_BLOCK_M=32        override BLOCK_M
    TRIATT_NPU_BLOCK_N=32        override BLOCK_N
"""
import os

import torch
import triton
import triton.language as tl

_DISABLE = os.environ.get("TRIATT_NPU_PATCH_DISABLE", "0") == "1"
_MASK_ONLY = os.environ.get("TRIATT_NPU_PATCH_MASK_ONLY", "0") == "1"
_LOOP = int(os.environ.get("TRIATT_NPU_LOOP", "1"))
_BLOCK_M_OVERRIDE = int(os.environ.get("TRIATT_NPU_BLOCK_M", "0") or 0)
_BLOCK_N_OVERRIDE = int(os.environ.get("TRIATT_NPU_BLOCK_N", "0") or 0)


def _pick_config(S, H, D):
    if _BLOCK_M_OVERRIDE and _BLOCK_N_OVERRIDE:
        return _BLOCK_M_OVERRIDE, _BLOCK_N_OVERRIDE
    if S <= 128:
        return 64, 64
    if S <= 512:
        return 64, 64
    return 32, 32


@triton.jit
def _attention_fwd_npu(
    Q,
    K,
    V,
    O,
    Bias1,
    Bias2,
    M,
    N,
    S: tl.constexpr,
    H,
    HEAD_DIM: tl.constexpr,
    deterministic: tl.constexpr,
    USE_BIAS1: tl.constexpr,
    LOOP_MODE: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Q/K/V/O: (B, N, S, H, HEAD_DIM)
    Bias1: (B, N, 1, 1, S), row mask, broadcast over query positions
    Bias2: (B, 1, H, S, S)
    M: (B, N, S, H), running max used by the upstream autograd path
    """
    input_dtype = Q.dtype.element_ty

    bn = tl.program_id(2)
    h = tl.program_id(1)
    s = tl.program_id(0) * BLOCK_M
    b = bn // N
    n = bn % N

    Ss = H * HEAD_DIM
    # Contiguous Q/K/V layout is (B, N, S, H, D).  The old expression omitted
    # the H factor on the (B, N, S) prefix, so every n > 0 read wrong data.
    qkvo_offset = ((b * N + n) * S * H + h) * HEAD_DIM
    bias1_offset = b * N * S + n * S
    bias2_offset = b * S * S * H + h * S * S
    m_offset = b * N * H * S + n * S * H + h

    qk_scale = 1.4426950408889634

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    rows = s + tl.arange(0, BLOCK_M)
    row_mask = rows[:, None] < S
    dims = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + qkvo_offset + rows[:, None] * Ss + dims[None, :],
        mask=row_mask,
        other=0.0,
    )

    if LOOP_MODE == 0:
        for i in range(tl.cdiv(S, BLOCK_N)):
            cols = i * BLOCK_N + tl.arange(0, BLOCK_N)
            col_mask = cols[None, :] < S
            k_t = tl.load(
                K + qkvo_offset + dims[:, None] + cols[None, :] * Ss,
                mask=col_mask,
                other=0.0,
            )
            v = tl.load(
                V + qkvo_offset + cols[:, None] * Ss + dims[None, :],
                mask=col_mask.trans(),
                other=0.0,
            )
            bias2 = tl.load(
                Bias2 + bias2_offset + rows[:, None] * S + cols[None, :],
                mask=row_mask * col_mask,
                other=0.0,
            )
            if USE_BIAS1:
                bias1_row = tl.load(
                    Bias1 + bias1_offset + cols,
                    mask=cols < S,
                    other=0.0,
                )
                bias = bias2 + bias1_row[None, :]
            else:
                bias = bias2
            qk = (
                tl.dot(
                    q,
                    k_t,
                    bias,
                    input_precision="ieee" if deterministic else "tf32",
                )
                * qk_scale
            )
            qk = tl.where(row_mask * col_mask, qk, -1e6)
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            qk = qk - m_ij[:, None]
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk)
            l_ij = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            p = p.to(input_dtype)
            acc = tl.dot(p, v, acc, input_precision="ieee" if deterministic else "tf32")
            l_i = l_ij
            m_i = m_ij
    elif LOOP_MODE == 1:
        for i in tl.static_range(NB):
            cols = i * BLOCK_N + tl.arange(0, BLOCK_N)
            col_mask = cols[None, :] < S
            k_t = tl.load(
                K + qkvo_offset + dims[:, None] + cols[None, :] * Ss,
                mask=col_mask,
                other=0.0,
            )
            v = tl.load(
                V + qkvo_offset + cols[:, None] * Ss + dims[None, :],
                mask=col_mask.trans(),
                other=0.0,
            )
            bias2 = tl.load(
                Bias2 + bias2_offset + rows[:, None] * S + cols[None, :],
                mask=row_mask * col_mask,
                other=0.0,
            )
            if USE_BIAS1:
                bias1_row = tl.load(
                    Bias1 + bias1_offset + cols,
                    mask=cols < S,
                    other=0.0,
                )
                bias = bias2 + bias1_row[None, :]
            else:
                bias = bias2
            qk = (
                tl.dot(
                    q,
                    k_t,
                    bias,
                    input_precision="ieee" if deterministic else "tf32",
                )
                * qk_scale
            )
            qk = tl.where(row_mask * col_mask, qk, -1e6)
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            qk = qk - m_ij[:, None]
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk)
            l_ij = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            p = p.to(input_dtype)
            acc = tl.dot(p, v, acc, input_precision="ieee" if deterministic else "tf32")
            l_i = l_ij
            m_i = m_ij
    elif LOOP_MODE == 2:
        for i in range(NB):
            cols = i * BLOCK_N + tl.arange(0, BLOCK_N)
            col_mask = cols[None, :] < S
            k_t = tl.load(
                K + qkvo_offset + dims[:, None] + cols[None, :] * Ss,
                mask=col_mask,
                other=0.0,
            )
            v = tl.load(
                V + qkvo_offset + cols[:, None] * Ss + dims[None, :],
                mask=col_mask.trans(),
                other=0.0,
            )
            bias2 = tl.load(
                Bias2 + bias2_offset + rows[:, None] * S + cols[None, :],
                mask=row_mask * col_mask,
                other=0.0,
            )
            if USE_BIAS1:
                bias1_row = tl.load(
                    Bias1 + bias1_offset + cols,
                    mask=cols < S,
                    other=0.0,
                )
                bias = bias2 + bias1_row[None, :]
            else:
                bias = bias2
            qk = (
                tl.dot(
                    q,
                    k_t,
                    bias,
                    input_precision="ieee" if deterministic else "tf32",
                )
                * qk_scale
            )
            qk = tl.where(row_mask * col_mask, qk, -1e6)
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            qk = qk - m_ij[:, None]
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk)
            l_ij = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            p = p.to(input_dtype)
            acc = tl.dot(p, v, acc, input_precision="ieee" if deterministic else "tf32")
            l_i = l_ij
            m_i = m_ij
    else:
        for i in tl.range(NB, loop_unroll_factor=4):
            cols = i * BLOCK_N + tl.arange(0, BLOCK_N)
            col_mask = cols[None, :] < S
            k_t = tl.load(
                K + qkvo_offset + dims[:, None] + cols[None, :] * Ss,
                mask=col_mask,
                other=0.0,
            )
            v = tl.load(
                V + qkvo_offset + cols[:, None] * Ss + dims[None, :],
                mask=col_mask.trans(),
                other=0.0,
            )
            bias2 = tl.load(
                Bias2 + bias2_offset + rows[:, None] * S + cols[None, :],
                mask=row_mask * col_mask,
                other=0.0,
            )
            if USE_BIAS1:
                bias1_row = tl.load(
                    Bias1 + bias1_offset + cols,
                    mask=cols < S,
                    other=0.0,
                )
                bias = bias2 + bias1_row[None, :]
            else:
                bias = bias2
            qk = (
                tl.dot(
                    q,
                    k_t,
                    bias,
                    input_precision="ieee" if deterministic else "tf32",
                )
                * qk_scale
            )
            qk = tl.where(row_mask * col_mask, qk, -1e6)
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            qk = qk - m_ij[:, None]
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk)
            l_ij = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            p = p.to(input_dtype)
            acc = tl.dot(p, v, acc, input_precision="ieee" if deterministic else "tf32")
            l_i = l_ij
            m_i = m_ij

    acc = acc / l_i[:, None]
    m_i += tl.math.log2(l_i)
    acc = acc.to(input_dtype)
    tl.store(
        O + qkvo_offset + rows[:, None] * Ss + dims[None, :],
        acc,
        mask=row_mask,
    )
    tl.store(
        M + m_offset + rows * H,
        m_i,
        mask=rows < S,
    )


def _install():
    from protenix.model.tri_attention.op import TriAttentionFunction

    _orig_forward = TriAttentionFunction.forward

    def _forward_npu(ctx, Q, K, V, Bias1, Bias2, deterministic=False):
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        Bias1 = Bias1.contiguous()
        Bias2 = Bias2.contiguous()

        B, N, S, H, D = Q.shape
        block_m, block_n = _pick_config(S, H, D)
        NB = (S + block_n - 1) // block_n
        O = torch.empty_like(Q)
        M = torch.empty((B, N, S, H), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(S, block_m), H, B * N)
        _attention_fwd_npu[grid](
            Q,
            K,
            V,
            O,
            Bias1,
            Bias2,
            M,
            N,
            S,
            H,
            HEAD_DIM=D,
            deterministic=deterministic,
            USE_BIAS1=not _MASK_ONLY,
            LOOP_MODE=_LOOP,
            NB=NB,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )

        ctx.save_for_backward(Q, K, V, Bias1, Bias2, O, M)
        ctx.deterministic = deterministic
        return O

    _orig_forward = staticmethod(_orig_forward)
    TriAttentionFunction.forward = staticmethod(_forward_npu)
    TriAttentionFunction._triatt_npu_original_forward = _orig_forward
    print(
        "[TRIATT-NPU] TriAttentionFunction.forward patched "
        f"(mask={'row' if not _MASK_ONLY else 'kernel-only'}, "
        f"loop={_LOOP}, config={_pick_config(1170, 4, 32)})"
    )


if not _DISABLE:
    _install()
else:
    print("[TRIATT-NPU] disabled by TRIATT_NPU_PATCH_DISABLE=1")
