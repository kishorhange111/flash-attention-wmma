"""FlashAttention-2 forward on Turing tensor cores (WMMA), built at first use.

from fawmma import flash_attention
o = flash_attention(q, k, v, causal=True)   # (B, H, N, 64) fp16 CUDA, N % 64 == 0
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

SRC = os.path.join(os.path.dirname(__file__), "flash_fwd.cu")


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    return load(
        name="fawmma_ext",
        sources=[SRC],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-arch=sm_75"],
    )


def flash_attention(q, k, v, causal: bool = False, return_lse: bool = False, padded: bool = True):
    """padded=False selects the v1 layout (64-element row strides, 16-way bank conflicts)."""
    o, lse = _ext().flash_fwd(q.contiguous(), k.contiguous(), v.contiguous(), causal, padded)
    return (o, lse) if return_lse else o


def extension_path() -> str:
    """Path of the compiled shared object, for inspecting its SASS."""
    import glob

    from torch.utils.cpp_extension import _get_build_directory

    _ext()
    return glob.glob(os.path.join(_get_build_directory("fawmma_ext", False), "*.so"))[0]


def reference(q, k, v, causal=False):
    """fp32 attention and its log-sum-exp, for checking."""
    s = (q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(
            torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf")
        )
    return s.softmax(-1) @ v.float(), torch.logsumexp(s, -1)
