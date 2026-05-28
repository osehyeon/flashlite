"""Batch FlashSigmoid (plan+run) vs torch reference over a ragged batch.

    .venv/bin/python flashlite/tests/test_flash_sigmoid.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashlite import flash_sigmoid_batch  # noqa: E402

RTOL, ATOL = 2e-2, 2e-2  # sigmoid (unnormalized) — looser, like single/flash_sigmoid


def torch_batch_sigmoid(q, k, v, qo_indptr, kv_indptr, logits_scale, sigmoid_bias):
    """sigmoid(QKᵀ·logits_scale + bias) @ V per sequence, no normalization."""
    outs = []
    B = qo_indptr.numel() - 1
    for i in range(B):
        qs, qe = int(qo_indptr[i]), int(qo_indptr[i + 1])
        ks, ke = int(kv_indptr[i]), int(kv_indptr[i + 1])
        qi, ki, vi = q[qs:qe], k[ks:ke], v[ks:ke]
        p = torch.sigmoid(
            torch.einsum("qhd,khd->hqk", qi.float(), ki.float()) * logits_scale + sigmoid_bias
        )
        outs.append(torch.einsum("hqk,khd->qhd", p, vi.float()).to(q.dtype))
    return torch.cat(outs, dim=0)


def main():
    torch.manual_seed(42)
    dev, dt = "cuda", torch.float16
    num_qo_heads = num_kv_heads = 8
    head_dim = 128
    logits_scale = 1.0 / math.sqrt(head_dim)
    sigmoid_bias = 0.25
    seq_lens = [20, 35, 16]
    B = len(seq_lens)

    indptr = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32, device=dev), 0, out=indptr[1:])
    total = int(indptr[-1])
    q = torch.randn(total, num_qo_heads, head_dim, dtype=dt, device=dev)
    k = torch.randn(total, num_kv_heads, head_dim, dtype=dt, device=dev)
    v = torch.randn(total, num_kv_heads, head_dim, dtype=dt, device=dev)

    o = flash_sigmoid_batch(q, k, v, indptr, indptr, num_qo_heads, num_kv_heads,
                            logits_scale, sigmoid_bias)
    o_ref = torch_batch_sigmoid(q, k, v, indptr, indptr, logits_scale, sigmoid_bias)

    abs_diff = (o.float() - o_ref.float()).abs()
    print(f"shape={tuple(o.shape)}  seqs={seq_lens}  "
          f"max|abs|={abs_diff.max().item():.3e}  mean={abs_diff.mean().item():.3e}")
    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)
    print(f"PASS: assert_close(rtol={RTOL}, atol={ATOL})")


if __name__ == "__main__":
    main()
