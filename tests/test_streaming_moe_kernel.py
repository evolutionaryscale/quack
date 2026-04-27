"""Stage 1 smoke test for the streaming-MoE kernel A scaffold.

Just verifies the kernel compiles and launches. No correctness check yet.
Stage 2 will add a real per-tile gemm + swiglu and compare to an eager
reference.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_streaming_moe_kernel_a_stub_launches(dtype):
    from quack.moe_streaming_sm90 import streaming_moe_kernel_a

    device = torch.device("cuda")
    T_recv, H, I, E_local = 32, 64, 32, 4
    TK = 16
    total_tiles = 4
    tile_m = 8

    recv_x = torch.randn(T_recv, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, H, 2 * I, dtype=dtype, device=device).mul_(0.02)
    postact_a = torch.zeros(TK, I, dtype=dtype, device=device)
    A_idx = torch.arange(TK, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device=device)
    cumulative_tiles_before_e = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    tile_records_expert_id = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

    streaming_moe_kernel_a(
        recv_x, W1, postact_a, A_idx,
        expert_frequency_offset, cumulative_tiles_before_e, tile_records_expert_id,
        total_tiles, tile_m,
    )
    torch.cuda.synchronize()

    assert postact_a[0, 0].item() == 1.0, "stub kernel did not write the sentinel"


if __name__ == "__main__":
    test_streaming_moe_kernel_a_stub_launches(torch.bfloat16)
    print("PASS")
