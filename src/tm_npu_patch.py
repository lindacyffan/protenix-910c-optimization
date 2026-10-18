"""
NPU-optimized TriangleMultiplicativeUpdate for Protenix on Ascend 910C.

Fuses 5 linear projections (a_g, a_p, b_g, b_p, g) that share the same
input (layer_norm_in(z)) into a single matmul. 100% numerically identical.

Optional: TM_PATCH_BF16=1 also keeps the bmm in bf16 instead of upcasting
to fp32 + std normalization. This is a bigger speedup but changes numerics
slightly — must verify lDDT doesn't drop.

Import AFTER npu_shim but BEFORE any protenix model code.

Env knobs:
    TM_PATCH_DISABLE=1   skip patching
    TM_PATCH_DEBUG=1     print shapes on first call
    TM_PATCH_BF16=1      keep bmm in bf16 (skip fp32 upcast + std norm)
"""
import os
import torch
import torch.nn.functional as F

_DISABLE = os.environ.get("TM_PATCH_DISABLE", "0") == "1"
_DEBUG = os.environ.get("TM_PATCH_DEBUG", "0") == "1"
_BF16_BMM = os.environ.get("TM_PATCH_BF16", "0") == "1"
_printed = set()

if _DISABLE:
    print("[TM PATCH] disabled by TM_PATCH_DISABLE=1")
else:
    import protenix.model.triangular.triangular as _tri
    from protenix.model.utils import is_fp16_enabled

    _orig_forward = _tri.TriangleMultiplicativeUpdate.forward

    def _call_original_forward(
        self,
        z,
        mask,
        inplace_safe,
        _add_with_inplace,
        _inplace_chunk_size,
        triangle_multiplicative,
    ):
        """Fallback without erasing the upstream argument defaults."""
        kwargs = {
            "inplace_safe": inplace_safe,
            "_add_with_inplace": _add_with_inplace,
            "triangle_multiplicative": triangle_multiplicative,
        }
        if _inplace_chunk_size is not None:
            kwargs["_inplace_chunk_size"] = _inplace_chunk_size
        return _orig_forward(self, z, mask=mask, **kwargs)

    def _ensure_fused_weights(self):
        """Lazily concatenate the 5 input-sharing linears into one weight.

        All of a_g, a_p, b_g, b_p, g take layer_norm_in(z) as input.
        F.linear(z, cat([W1..W5], dim=0)) is bit-identical to running
        each linear separately, because each output element's dot product
        over the input dimension is independent.
        """
        if getattr(self, "_tm_fused", False):
            return
        with torch.no_grad():
            Wa_g = self.linear_a_g.weight  # (c_hidden, c_z)
            Wa_p = self.linear_a_p.weight
            Wb_g = self.linear_b_g.weight
            Wb_p = self.linear_b_p.weight
            Wg = self.linear_g.weight       # (c_z, c_z)
            self._W_in_fused = torch.cat(
                [Wa_g, Wa_p, Wb_g, Wb_p, Wg], dim=0
            ).contiguous()
        self._tm_fused = True

    def _tm_forward_npu(
        self,
        z,
        mask=None,
        inplace_safe=False,
        _add_with_inplace=False,
        _inplace_chunk_size=None,
        triangle_multiplicative="torch",
    ):
        # Only override the plain "torch" + non-inplace path.
        # cuequivariance path and inplace_safe path fall through to original.
        if triangle_multiplicative == "cuequivariance" and self.c_z == self.c_hidden:
            return _call_original_forward(
                self,
                z,
                mask,
                inplace_safe,
                _add_with_inplace,
                _inplace_chunk_size,
                triangle_multiplicative,
            )
        if inplace_safe:
            return _call_original_forward(
                self,
                z,
                mask,
                inplace_safe,
                _add_with_inplace,
                _inplace_chunk_size,
                triangle_multiplicative,
            )

        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        _input_inplace_safe = inplace_safe is True
        if _input_inplace_safe and _add_with_inplace:
            z_in = z.clone()

        # --- LayerNorm input (same as original) ---
        z = self.layer_norm_in(z)

        # --- Fused projection: 5 linears -> 1 matmul ---
        _ensure_fused_weights(self)
        ch = self.c_hidden
        cz = self.c_z
        proj = F.linear(z, self._W_in_fused)   # (..., N, N, 4*ch + cz)
        a_g = proj[..., 0:ch]
        a_p = proj[..., ch:2 * ch]
        b_g = proj[..., 2 * ch:3 * ch]
        b_p = proj[..., 3 * ch:4 * ch]
        g_raw = proj[..., 4 * ch:4 * ch + cz]

        # --- Gating + projection (same math as original) ---
        a = mask * torch.sigmoid(a_g) * a_p
        b = mask * torch.sigmoid(b_g) * b_p

        if _DEBUG and self.__class__.__name__ not in _printed:
            _printed.add(self.__class__.__name__)
            print(f"[TM PATCH] {self.__class__.__name__} "
                  f"z={list(z.shape)} proj={list(proj.shape)} "
                  f"ch={ch} cz={cz} bf16_bmm={_BF16_BMM} "
                  f"fp16_enabled={is_fp16_enabled()}")

        # --- Combine projections (the O(N^3) bmm) ---
        if _BF16_BMM:
            # v2: keep bmm in bf16, skip std normalization
            x = self._combine_projections(a, b)
        elif is_fp16_enabled():
            # original: std-normalize then upcast to fp32 for bmm
            a_std = a.std()
            b_std = b.std()
            if a_std != 0.0 and b_std != 0.0:
                a = a / a_std
                b = b / b_std
            with torch.amp.autocast("cuda", enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        x = x * torch.sigmoid(g_raw)

        if _input_inplace_safe and _add_with_inplace:
            x = x + z_in
        return x

    _tri.TriangleMultiplicativeUpdate.forward = _tm_forward_npu
    print(f"[TM PATCH] TriangleMultiplicativeUpdate.forward patched "
          f"(bf16_bmm={'ON' if _BF16_BMM else 'OFF'})")
