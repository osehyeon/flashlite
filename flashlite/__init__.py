"""flashlite — a lightweight, importable port of the FlashInfer study variants.

Currently exposes two batch (ragged plan+run) attention variants built on the
FlashInfer FA2 template path:

    from flashlite import flash_attention2_batch, flash_sigmoid_batch
"""

from .flash_attention2 import flash_attention2_batch
from .flash_sigmoid import flash_sigmoid_batch

__all__ = ["flash_attention2_batch", "flash_sigmoid_batch"]
