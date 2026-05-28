"""FlashAttention-2 (standard softmax) — BATCH ragged plan+run.

Ported from the study repo ``custom/batch/flash_attention2``. Uses FlashInfer's
C++ template-variant JIT path on the FA2 backend (the only backend available on
sm_120 / RTX 5090). The variant struct below spells out all four
``AttentionVariantBase`` hooks at their base defaults — standard softmax needs
none of them overridden, so this doubles as a reference for "what each hook's
default is".

Ragged = dense K/V but batched: q is [Σ q_len, Hq, D], k/v are [Σ kv_len, Hkv, D],
delimited by ``qo_indptr`` / ``kv_indptr`` (cumulative lengths). sm_scale is an
additional scalar passed through ``run()``.
"""

from __future__ import annotations

import math

import torch

WORKSPACE_BYTES = 128 * 1024 * 1024

STANDARD_ATTENTION_FA2_DECL = r"""
struct StandardAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ StandardAttention(const Params& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  // --- All 4 hooks written out verbatim as the AttentionVariantBase defaults.
  // Standard softmax needs none of them overridden; they are spelled out here
  // only as a reference for "what is the base default for each hook".

  // ① LogitsTransform — base default: identity.
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  });

  // ② LogitsMask — base default: keep every (qo, kv) pair (no masking).
  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return true;
  });

  // ③ update_m_d — base default: no-op (no virtual token injected into m/d).
  REGISTER_M_D_UPDATE(params, kv_tile_idx, qo_head_idx, m, d, scale, {
    return;
  });

  // ④ OutputTransform — base default: softmax normalization output * 1/d * v_scale.
  //   d   : row-sum of the exp2 weights (softmax denominator)
  //   m   : row-max; m == -inf marks a fully-masked row -> emit 0
  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
    float v_scale_val = get_v_scale(params);
    return output * d_rcp * v_scale_val;
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
            f"batch_prefill_standard_attn_d{head_dim}",  # uri
            dtype, dtype, dtype,        # dtype_q, dtype_kv, dtype_o
            torch.int32,                # idtype (indptr dtype)
            head_dim, head_dim,         # head_dim_qk, head_dim_vo
            [], [],                     # additional_tensor names/dtypes
            ["sm_scale"], ["double"],   # additional_scalar names/dtypes
            "StandardAttention", STANDARD_ATTENTION_FA2_DECL,
        )
        _wrappers[key] = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _workspace, kv_layout="NHD", backend="fa2", jit_args=jit_args,
        )
    return _wrappers[key]


def flash_attention2_batch(q, k, v, qo_indptr, kv_indptr,
                           num_qo_heads, num_kv_heads, sm_scale=None, causal=False):
    """q: [Σq, Hq, D]; k, v: [Σkv, Hkv, D]; qo_indptr/kv_indptr: [B+1] int32."""
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    w = _get_wrapper(head_dim, q.dtype, q.device)
    w.plan(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim,
           causal=causal, q_data_type=q.dtype)
    return w.run(q, k, v, sm_scale)
