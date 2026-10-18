# Protenix Ascend 910C Optimization

This repository contains the NPU attention patches and evaluation workflow used
to run Protenix on Ascend 910C with an FP32 storage path.

## Optimization path

- `src/pse_patch_v1.py` routes TriangleAttention to NPU fusion attention with a position-specific scale/bias tensor and a boolean row mask.
- `src/apb_pse_patch_v2.py` routes the 4D and 3D attention primitives used by atom-pair bias to NPU flash attention.
- `src/triatt_npu_patch.py` adapts TriangleAttention layouts, row masking, and loop tiling for NPU execution.
- `src/tm_npu_patch.py` fuses parts of TriangleMultiplicativeUpdate while keeping BF16 matrix multiplication disabled for the FP32 comparison path.
- `src/flagos_910c_launcher.py` installs the selected patches before the Protenix CLI starts.

The implementation intentionally keeps model parameters in FP32 and disables TF32, AMP mixed precision, and shared diffusion caching for controlled cross-platform comparisons.

## Same-workload result

Both runs used protein `7WUX`, seed `101`, `N_msa=1302`, `N_token=1170`, `N_atom=9094`, cycle `10`, 200 diffusion steps, one sample, FP32 storage, TF32 disabled, and caching disabled.

| Route | Forward | Total | pLDDT | pTM | ipTM | Ranking |
|---|---:|---:|---:|---:|---:|---:|
| H100 native FP32, TF32 off | 116.53 s | 124.31 s | 90.6176 | 0.925347 | 0.929381 | 0.928575 |
| Ascend 910C + PSE | 368.39 s | 378.99 s | 90.3770 | 0.924468 | 0.928870 | 0.927990 |

Relative confidence differences were below 0.27%, satisfying the 5% target. The 910C timing was collected while the NPU already had about 29 GB of HBM allocated to another workload, so it is a resource-constrained diagnostic measurement rather than a clean-device performance result.

## Reproduce

Copy the files from `src/` into a Protenix checkout, then run:

```bash
PROTENIX_ROOT=/path/to/Protenix \
NPU_ID=8 \
OUTPUT_NAME=output_910c_same_msa_fp32_tf32off_pse \
bash benchmarks/run_same_config_pse.sh
```

The script expects the CANN 8.5 environment and the Protenix checkpoint and input file to already be installed. Use an idle NPU for performance measurements.
