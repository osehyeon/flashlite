"""FlashSigmoid — BATCH ragged plan+run.

Ported from the study repo ``custom/batch/flash_sigmoid``. Same FA2 template path
as :mod:`flashlite.flash_attention2`, but the variant swaps softmax for a sigmoid
activation and disables normalization (``use_softmax = false``):

    O = sigmoid(QKᵀ·logits_scale + sigmoid_bias) @ V          (no 1/d)

``logits_scale`` + ``sigmoid_bias`` are additional scalars passed through
``run(q, k, v, logits_scale, sigmoid_bias)``. Mirrors flashinfer's official
``test_batch_prefill_flash_sigmoid``.
"""

from __future__ import annotations

import math

import torch

WORKSPACE_BYTES = 128 * 1024 * 1024

FLASH_SIGMOID_FA2_DECL = r"""
struct FlashSigmoid : AttentionVariantBase {
  static constexpr bool use_softmax = false;

  uint32_t window_left, qo_len, kv_len;
  float sigmoid_scale_log2;
  float sigmoid_bias_log2;

  template <typename Params>
  __device__ __host__ FlashSigmoid(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sigmoid_bias_log2 = params.sigmoid_bias * math::log2e;
    sigmoid_scale_log2 = params.logits_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits * sigmoid_scale_log2 + sigmoid_bias_log2)));
  });

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    return output;
  })
};
"""

_workspace: torch.Tensor | None = None
_wrappers: dict = {}


def _get_wrapper(head_dim: int, dtype: torch.dtype, device):
    global _workspace
    import flashinfer
    if _workspace is None:
        _workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    key = (head_dim, dtype)
    if key not in _wrappers:
        jit_args = (
            f"batch_prefill_flash_sigmoid_d{head_dim}",  # uri
            dtype, dtype, dtype,
            torch.int32,
            head_dim, head_dim,
            [], [],
            ["logits_scale", "sigmoid_bias"], ["double", "double"],
            "FlashSigmoid", FLASH_SIGMOID_FA2_DECL,
        )
        _wrappers[key] = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _workspace, kv_layout="NHD", backend="fa2", jit_args=jit_args,
        )
    return _wrappers[key]


def flash_sigmoid_batch(q, k, v, qo_indptr, kv_indptr,
                        num_qo_heads, num_kv_heads, logits_scale=None, sigmoid_bias=0.0):
    """q: [Σq, Hq, D]; k, v: [Σkv, Hkv, D]; qo_indptr/kv_indptr: [B+1] int32."""
    head_dim = q.shape[-1]
    if logits_scale is None:
        logits_scale = 1.0 / math.sqrt(head_dim)
    w = _get_wrapper(head_dim, q.dtype, q.device)
    # causal=False: sigmoid attention is unnormalized & non-causal here.
    w.plan(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim,
           causal=False, q_data_type=q.dtype)
    return w.run(q, k, v, logits_scale, sigmoid_bias)
