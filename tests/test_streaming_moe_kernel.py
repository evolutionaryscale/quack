"""Tests for the streaming-MoE kernel A.

These tests exercise the streaming-design properties directly:
  - test_compile: kernel compiles for a representative shape.
  - test_single_tile: total_tiles=1, tile_ready pre-set, validates IP1+IP2+IP3
    wiring + per-tile spin-then-read. Numerics: gather + matmul + SwiGLU
    against an eager pytorch reference.
  - test_multi_tile_static: total_tiles=N, all tile_ready slots pre-set,
    persistent CTAs absorb all tiles.
  - test_producer_consumer: producer kernel on a different stream fires
    tile_ready entries with delay; kernel A spins then drains.

Linear-claim layout:
  * tile_ready[total_tiles] int64 — release stamps from slot_assign (or test
    stub). Consumer spins until tile_ready[tile_id] >= dispatch_seq.
  * consumer_head[1] int32 — single global atomic-add counter for linear claims.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _swiglu_ref(h_two_I: torch.Tensor) -> torch.Tensor:
    """QuACK's gated epilogue pairs ADJACENT N-elements: gate = h[..., ::2],
    up = h[..., 1::2], output = silu(gate) * up. See gemm_act.py
    GemmGatedMixin.epi_visit_subtile.
    """
    gate = h_two_I[..., 0::2]
    up = h_two_I[..., 1::2]
    return F.silu(gate) * up


def _make_tile_ready(total_tiles: int, dispatch_seq: int, device, fired: bool = True) -> torch.Tensor:
    """Allocate tile_ready[total_tiles] int64. If fired=True, pre-set to
    dispatch_seq (all tiles ready at launch); else zero (test producer fires)."""
    val = dispatch_seq if fired else 0
    return torch.full((total_tiles,), val, dtype=torch.int64, device=device)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_streaming_moe_a_compiles(device):
    """JIT-compile only (no launch) for a representative production-shape config."""
    from quack.moe_streaming_sm90 import streaming_moe_a

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    total_tiles = 4
    T_recv = total_tiles * tile_m

    dtype = torch.bfloat16
    recv_x = torch.randn(T_recv, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)
    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)

    tile_records_recv_x_rows = (
        torch.arange(total_tiles * tile_m, dtype=torch.int32, device=device)
        .view(total_tiles, tile_m)
    )
    tile_records_expert_id = torch.zeros(total_tiles, dtype=torch.int32, device=device)
    tile_ready = _make_tile_ready(total_tiles, dispatch_seq=1, device=device)
    consumer_head = torch.zeros(1, dtype=torch.int32, device=device)

    import quack.cache_utils as cu
    orig = cu.COMPILE_ONLY
    cu.COMPILE_ONLY = True
    try:
        streaming_moe_a(
            recv_x, W1, postact_a,
            tile_records_recv_x_rows, tile_records_expert_id,
            tile_ready, consumer_head,
            dispatch_seq=1,
            tile_m=tile_m, tile_n=tile_n,
        )
    finally:
        cu.COMPILE_ONLY = orig


def test_streaming_moe_a_single_tile(device):
    """total_tiles=1, tile_ready pre-set. Validates the full kernel path:
    linear claim, scheduler 5-int payload, per-tile gather, per-tile postact,
    expert lookup via tile_records_expert_id.
    """
    from quack.moe_streaming_sm90 import streaming_moe_a

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    total_tiles = 1
    T_recv = 256

    dtype = torch.bfloat16
    torch.manual_seed(0)
    recv_x = torch.randn(T_recv, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)

    chosen_expert = 2
    chosen_rows = torch.randperm(T_recv, device=device)[:tile_m].to(torch.int32)

    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    tile_records_recv_x_rows = chosen_rows.view(1, tile_m).contiguous()
    tile_records_expert_id = torch.tensor([chosen_expert], dtype=torch.int32, device=device)

    tile_ready = _make_tile_ready(total_tiles, dispatch_seq=1, device=device)
    consumer_head = torch.zeros(1, dtype=torch.int32, device=device)

    streaming_moe_a(
        recv_x, W1, postact_a,
        tile_records_recv_x_rows, tile_records_expert_id,
        tile_ready, consumer_head,
        dispatch_seq=1,
        tile_m=tile_m, tile_n=tile_n,
    )
    torch.cuda.synchronize()

    x_gathered = recv_x[chosen_rows.long()]
    h = x_gathered.float() @ W1[chosen_expert].float().t()
    a_ref = _swiglu_ref(h).to(dtype)

    a_kernel = postact_a[0]
    diff = (a_kernel.float() - a_ref.float()).abs()
    rel = diff / (a_ref.float().abs() + 1e-3)
    assert rel.max().item() < 5e-2, (
        f"max rel diff {rel.max().item():.4f}, max abs diff {diff.max().item():.4f}"
    )


def test_streaming_moe_a_multi_tile_static(device):
    """total_tiles=N>1 spread across multiple experts. Validates per-tile
    expert selection (W1[expert_id] varies), per-tile gather indices, and
    persistent kernel termination via the linear-claim bounds check.
    """
    from quack.moe_streaming_sm90 import streaming_moe_a

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    total_tiles = 6
    T_recv = 1024

    dtype = torch.bfloat16
    torch.manual_seed(7)
    recv_x = torch.randn(T_recv, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)

    tile_records_expert_id = torch.tensor(
        [t % E_local for t in range(total_tiles)], dtype=torch.int32, device=device
    )
    tile_records_recv_x_rows = torch.empty(total_tiles, tile_m, dtype=torch.int32, device=device)
    for t in range(total_tiles):
        g = torch.Generator(device=device).manual_seed(100 + t)
        tile_records_recv_x_rows[t] = torch.randperm(T_recv, generator=g, device=device)[:tile_m].to(torch.int32)

    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    tile_ready = _make_tile_ready(total_tiles, dispatch_seq=1, device=device)
    consumer_head = torch.zeros(1, dtype=torch.int32, device=device)

    streaming_moe_a(
        recv_x, W1, postact_a,
        tile_records_recv_x_rows, tile_records_expert_id,
        tile_ready, consumer_head,
        dispatch_seq=1,
        tile_m=tile_m, tile_n=tile_n,
    )
    torch.cuda.synchronize()

    for t in range(total_tiles):
        e = tile_records_expert_id[t].item()
        rows = tile_records_recv_x_rows[t].long()
        x_gathered = recv_x[rows]
        h = x_gathered.float() @ W1[e].float().t()
        a_ref = _swiglu_ref(h).to(dtype)
        diff = (postact_a[t].float() - a_ref.float()).abs()
        rel = diff / (a_ref.float().abs() + 1e-3)
        assert rel.max().item() < 5e-2, (
            f"tile {t}: expert={e}, max rel diff {rel.max().item():.4f}, "
            f"max abs diff {diff.max().item():.4f}"
        )


def test_streaming_moe_a_producer_consumer(device):
    """Kernel A on compute_a_stream spins on tile_ready while a producer
    kernel on a separate stream release-stores dispatch_seq slot by slot
    with delays between fires.
    """
    from quack.moe_streaming_sm90 import streaming_moe_a, fire_tiles_with_delay

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    total_tiles = 6
    T_recv = 1024

    dtype = torch.bfloat16
    torch.manual_seed(11)
    recv_x = torch.randn(T_recv, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)

    tile_records_expert_id = torch.tensor(
        [t % E_local for t in range(total_tiles)], dtype=torch.int32, device=device
    )
    tile_records_recv_x_rows = torch.empty(total_tiles, tile_m, dtype=torch.int32, device=device)
    for t in range(total_tiles):
        g = torch.Generator(device=device).manual_seed(200 + t)
        tile_records_recv_x_rows[t] = torch.randperm(T_recv, generator=g, device=device)[:tile_m].to(torch.int32)

    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    tile_ready = _make_tile_ready(total_tiles, dispatch_seq=1, device=device, fired=False)
    consumer_head = torch.zeros(1, dtype=torch.int32, device=device)

    # Pre-warm the producer JIT compile so the host doesn't block during the
    # concurrent launch (use dispatch_seq=999 then reset).
    fire_tiles_with_delay(tile_ready, dispatch_seq=999, delay_us=0)
    torch.cuda.synchronize()
    tile_ready.zero_()
    torch.cuda.synchronize()

    compute_a_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()

    with torch.cuda.stream(compute_a_stream):
        streaming_moe_a(
            recv_x, W1, postact_a,
            tile_records_recv_x_rows, tile_records_expert_id,
            tile_ready, consumer_head,
            dispatch_seq=1,
            tile_m=tile_m, tile_n=tile_n,
        )
    with torch.cuda.stream(producer_stream):
        fire_tiles_with_delay(tile_ready, dispatch_seq=1, delay_us=50)

    torch.cuda.synchronize()

    for t in range(total_tiles):
        e = tile_records_expert_id[t].item()
        rows = tile_records_recv_x_rows[t].long()
        x_gathered = recv_x[rows]
        h = x_gathered.float() @ W1[e].float().t()
        a_ref = _swiglu_ref(h).to(dtype)
        diff = (postact_a[t].float() - a_ref.float()).abs()
        rel = diff / (a_ref.float().abs() + 1e-3)
        assert rel.max().item() < 5e-2, (
            f"tile {t}: expert={e}, max rel diff {rel.max().item():.4f}, "
            f"max abs diff {diff.max().item():.4f}"
        )


if __name__ == "__main__":
    dev = torch.device("cuda")
    test_streaming_moe_a_compiles(dev)
    print("compile OK")
    test_streaming_moe_a_single_tile(dev)
    print("single-tile PASS")
    test_streaming_moe_a_multi_tile_static(dev)
    print("multi-tile-static PASS")
    test_streaming_moe_a_producer_consumer(dev)
    print("producer-consumer PASS")
