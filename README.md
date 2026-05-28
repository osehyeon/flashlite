# flashlite

Importable port of FlashInfer study variants. Uses the FA2 template JIT path (the only backend that runs on sm_120 / RTX 5090).

## Variants (batch, ragged plan+run)

- `flash_attention2_batch` — standard softmax
- `flash_sigmoid_batch` — `sigmoid(QKᵀ·scale + bias) @ V`, no normalization

## Usage

```python
import flashlite

o = flashlite.flash_attention2_batch(q, k, v, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, sm_scale)
o = flashlite.flash_sigmoid_batch(q, k, v, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, logits_scale, sigmoid_bias)
```

## Test

```bash
python flashlite/tests/test_flash_attention2.py
python flashlite/tests/test_flash_sigmoid.py
```

Requires `flashinfer-python`, PyTorch, CUDA ≥ 12.9.
