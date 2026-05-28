# flashlite

FlashInfer 학습 변형들을 import 가능한 경량 패키지로 포팅한 것.
FlashInfer의 **FA2 템플릿 JIT 경로**를 백엔드로 사용한다 (sm_120 / RTX 5090에서 동작하는 유일한 backend).

## 제공 변형 (batch, ragged plan+run)

| 함수 | 설명 |
|---|---|
| `flash_attention2_batch` | 표준 softmax. DECL에 4개 훅(LogitsTransform / LogitsMask / update_m_d / OutputTransform)을 기본값 그대로 명시 |
| `flash_sigmoid_batch` | `sigmoid(QKᵀ·scale + bias) @ V`, 정규화 없음 (`use_softmax = false`) |

## 사용

```python
import flashlite

# q, k: [Σ q_len, H, D] / [Σ kv_len, H, D],  indptr: [B+1] int32
o = flashlite.flash_attention2_batch(q, k, v, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, sm_scale)
o = flashlite.flash_sigmoid_batch(q, k, v, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, logits_scale, sigmoid_bias)
```

## 테스트

각 테스트는 torch 레퍼런스와 `torch.testing.assert_close`로 일치를 검증한다.

```bash
env CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PWD/.venv/bin:$PATH \
    python flashlite/tests/test_flash_attention2.py
env CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PWD/.venv/bin:$PATH \
    python flashlite/tests/test_flash_sigmoid.py
```

## 요구사항

- FlashInfer (`flashinfer-python`), PyTorch, CUDA toolkit ≥ 12.9 (sm_120 지원). 본 환경은 13.0 사용.
