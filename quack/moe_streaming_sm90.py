"""Streaming-MoE kernel A (CuTeDSL, SM90).

Implements the kernel A stage of the problem-tile streaming pipeline (see
`new_design.md` §"Pipeline architecture"): pulls problem tiles from a
producer-fed queue, gathers the corresponding `recv_x` rows, runs gemm against
`W1[expert_id]`, applies SwiGLU, and stores the I-half output into `postact_a`.

This file is built progressively (see logbook):
  Stage 1 — scaffold + stub kernel (this commit)
  Stage 2 — single-tile gather + matmul + swiglu (Python-driven loop)
  Stage 3 — persistent kernel, static partition
  Stage 4 — queue-pull (atomicAdd + spin on tile_ready_queue_seq)
  Stage 5 — sticky-CTA-to-expert
  Stage 6 — push to a_ready_queue for kernel Y
"""

from __future__ import annotations

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr

from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map


class StreamingMoeSm90:
    """Streaming-MoE kernel A (forward, SM90).

    Stage 1: stub. The kernel currently writes a sentinel into `postact_a[0]`
    so we can verify the dispatch path end-to-end before adding real compute.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        H: int,
        I: int,
        E_local: int,
        tile_m: int = 128,
    ):
        self.dtype = dtype
        self.H = H
        self.I = I
        self.E_local = E_local
        self.tile_m = tile_m

    @cute.jit
    def __call__(
        self,
        recv_x: cute.Tensor,                       # [T_recv, H]
        W1: cute.Tensor,                           # [E_local, H, 2*I]  (contracts H)
        postact_a: cute.Tensor,                    # [TK, I]
        A_idx: cute.Tensor,                        # [TK] int32, gather indices into recv_x
        expert_frequency_offset: cute.Tensor,      # [E_local + 1] int32
        cumulative_tiles_before_e: cute.Tensor,    # [E_local + 1] int32
        tile_records_expert_id: cute.Tensor,       # [total_tiles] int32
        total_tiles: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # Stage 1: launch stub. Single block, single thread, writes a sentinel.
        self.kernel(postact_a).launch(
            grid=[1, 1, 1],
            block=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, postact_a: cute.Tensor):
        # Stage 1: writes a sentinel value into the output to prove the kernel ran.
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            # Touch the first element so the stub has an observable side effect.
            postact_a[0, 0] = postact_a.element_type(1)


@jit_cache
def _compile_streaming_moe_fwd(dtype: Type[cutlass.Numeric], H: int, I: int, E_local: int, tile_m: int):
    T_recv_sym = cute.sym_int()
    TK_sym = cute.sym_int()
    total_tiles_sym = cute.sym_int()

    recv_x = fake_tensor(dtype, (T_recv_sym, H), divisibility=8)
    W1 = fake_tensor(dtype, (E_local, H, 2 * I), divisibility=8)
    postact_a = fake_tensor(dtype, (TK_sym, I), divisibility=8)
    A_idx = fake_tensor(cutlass.Int32, (TK_sym,), divisibility=1)
    efo = fake_tensor(cutlass.Int32, (E_local + 1,), divisibility=1)
    ctbe = fake_tensor(cutlass.Int32, (E_local + 1,), divisibility=1)
    expert_id = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)

    op = StreamingMoeSm90(dtype, H, I, E_local, tile_m)
    return cute.compile(
        op,
        recv_x, W1, postact_a, A_idx, efo, ctbe, expert_id,
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@torch.library.custom_op("quack::streaming_moe_kernel_a", mutates_args={"postact_a"})
def streaming_moe_kernel_a(
    recv_x: torch.Tensor,
    W1: torch.Tensor,
    postact_a: torch.Tensor,
    A_idx: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    cumulative_tiles_before_e: torch.Tensor,
    tile_records_expert_id: torch.Tensor,
    total_tiles: int,
    tile_m: int,
) -> None:
    """Run the streaming-MoE kernel A in place into ``postact_a``."""
    assert recv_x.is_cuda and W1.is_cuda and postact_a.is_cuda
    assert recv_x.dim() == 2 and W1.dim() == 3 and postact_a.dim() == 2
    H = recv_x.size(1)
    E_local = W1.size(0)
    Two_I = W1.size(2)
    assert Two_I % 2 == 0
    I = Two_I // 2
    dtype = torch2cute_dtype_map[recv_x.dtype]
    _compile_streaming_moe_fwd(dtype, H, I, E_local, tile_m)(
        recv_x, W1, postact_a, A_idx,
        expert_frequency_offset, cumulative_tiles_before_e, tile_records_expert_id,
        total_tiles,
    )


@streaming_moe_kernel_a.register_fake
def _streaming_moe_kernel_a_fake(
    recv_x: torch.Tensor,
    W1: torch.Tensor,
    postact_a: torch.Tensor,
    A_idx: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    cumulative_tiles_before_e: torch.Tensor,
    tile_records_expert_id: torch.Tensor,
    total_tiles: int,
    tile_m: int,
) -> None:
    pass
