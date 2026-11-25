"""GPU tests: they run on Colab (T4). On machines without CUDA they are skipped, and CI
checks only that the CPU reference is right."""

import pytest
import torch
import torch.nn.functional as F

from fawmma import reference

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_reference_matches_sdpa_on_cpu():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 3, 128, 64, generator=g) for _ in range(3))
    for causal in (False, True):
        o, lse = reference(q, k, v, causal)
        torch.testing.assert_close(
            o, F.scaled_dot_product_attention(q, k, v, is_causal=causal), atol=1e-5, rtol=1e-5
        )
        assert lse.shape == (2, 3, 128)


@cuda
@pytest.mark.parametrize("padded", [True, False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 64), (2, 4, 256), (1, 8, 1024), (3, 2, 192)])
def test_matches_fp32_reference(shape, causal, padded):
    from fawmma import flash_attention

    torch.manual_seed(0)
    b, h, n = shape
    q, k, v = (torch.randn(b, h, n, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    o, lse = flash_attention(q, k, v, causal, return_lse=True, padded=padded)
    ro, rlse = reference(q, k, v, causal)
    torch.testing.assert_close(o.float(), ro, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(lse, rlse, atol=1e-3, rtol=1e-3)


@cuda
def test_large_logits_do_not_overflow():
    """Online softmax subtracts the running max: scores of ~±500 must not produce inf/NaN."""
    from fawmma import flash_attention

    torch.manual_seed(1)
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) * 8
    k = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) * 8
    v = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16)
    o = flash_attention(q, k, v)
    assert torch.isfinite(o).all()
    torch.testing.assert_close(o.float(), reference(q, k, v)[0], atol=5e-3, rtol=5e-3)


@cuda
def test_rejects_unsupported_shapes():
    from fawmma import flash_attention

    x = torch.randn(1, 1, 100, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        flash_attention(x, x, x)
    y = torch.randn(1, 1, 128, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        flash_attention(y, y, y)
