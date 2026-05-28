"""Batch FlashAttention-2 (plan+run) vs torch reference over a ragged batch.

    .venv/bin/python flashlite/tests/test_flash_attention2.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashlite import flash_attention2_batch  # noqa: E402

RTOL, ATOL = 1e-3, 1e-3


def torch_batch_softmax(q, k, v, qo_indptr, kv_indptr, sm_scale):
    """Loop over sequences delimited by indptr, full (non-causal) softmax each."""
    outs = []
    B = qo_indptr.numel() - 1
    for i in range(B):
        qs, qe = int(qo_indptr[i]), int(qo_indptr[i + 1])
        ks, ke = int(kv_indptr[i]), int(kv_indptr[i + 1])
        qi, ki, vi = q[qs:qe], k[ks:ke], v[ks:ke]
        logits = torch.einsum("qhd,khd->hqk", qi.float(), ki.float()) * sm_scale
        p = torch.softmax(logits, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vi.float()).to(q.dtype))
    return torch.cat(outs, dim=0)


def main():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.float16
    num_qo_heads = num_kv_heads = 8
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    seq_lens = [20, 35, 16]            # q_len == kv_len per sequence (self-attn prefill)
    B = len(seq_lens)

    indptr = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32, device=dev), 0, out=indptr[1:])
    total = int(indptr[-1])
    q = torch.randn(total, num_qo_heads, head_dim, dtype=dt, device=dev)
    k = torch.randn(total, num_kv_heads, head_dim, dtype=dt, device=dev)
    v = torch.randn(total, num_kv_heads, head_dim, dtype=dt, device=dev)

    o = flash_attention2_batch(q, k, v, indptr, indptr, num_qo_heads, num_kv_heads, sm_scale)
    o_ref = torch_batch_softmax(q, k, v, indptr, indptr, sm_scale)

    abs_diff = (o.float() - o_ref.float()).abs()
    print(f"shape={tuple(o.shape)}  seqs={seq_lens}  "
          f"max|abs|={abs_diff.max().item():.3e}  mean={abs_diff.mean().item():.3e}")
    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)
    print(f"PASS: assert_close(rtol={RTOL}, atol={ATOL})")


if __name__ == "__main__":
    main()
