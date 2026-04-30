"""Streaming-MoE kernel A (CuTeDSL, SM90, pool layout).

Forward kernel A of the problem-tile streaming pipeline:
  * Persistent CTAs pull tiles from a producer-fed queue (`tile_ready`).
  * For each claimed tile_id, the scheduler reads `expert_id =
    tile_id_to_expert[tile_id]` and computes `pid_m = tile_id -
    expert_pool_block_offset[expert_id]`.
  * Standard varlen_m strided TMA load of `pool[tile_id * tile_M : ..., :]`
    (the row offset = `cu_seqlens_m[expert_id] + pid_m * tile_m` lands at
    the correct expert-major pool row by construction).
  * GEMM against W1[expert_id], SwiGLU register-resident epilogue, TMA-store
    the I-half post-activation to `postact_a[tile_id * tile_M : ..., :]`.

Inherits the GEMM mainloop, SwiGLU epilogue, scheduler-warp + pipeline-state
machinery from `quack.gemm_act.GemmGatedSm90`. Streaming-specific behavior is
isolated to three overrides:
  (1) get_scheduler_class — return StreamingTileScheduler.
  (2) get_scheduler_arguments — build StreamingTileSchedulerArguments from
      pool-shape metadata.
  (3) epi_setup_postact — postact destination indexed by tile_id, not by an
      mAIdx-derived varlen offset.

The `sched_payload_ints = 5` constructor field bumps the scheduler-to-consumer
SMEM payload to carry tile_id alongside (pid_m, pid_n, batch_idx, is_valid)
for the postact destination override.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from quack import copy_utils
from quack.activation import gate_fn_map
from quack.cache_utils import jit_cache, COMPILE_ONLY
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.gemm_act import GemmGatedSm90
from quack.gemm_tvm_ffi_utils import compile_gemm_kernel
from quack.rounding import RoundingMode
from quack.tile_scheduler import (
    PersistenceMode,
    StreamingTileScheduler,
    StreamingTileSchedulerArguments,
)
from quack.varlen_utils import VarlenArguments


# ---------------------------------------------------------------------------
# Host-facing scheduler-options NamedTuple. Mirrors TileSchedulerOptions but
# carries the streaming-specific tensors/pointers that the scheduler needs.
# ---------------------------------------------------------------------------
@mlir_namedtuple
class StreamingTileSchedulerOptions(NamedTuple):
    max_active_clusters: Int32
    consumer_head: cute.Tensor                 # [1] int32 — global linear claim counter
    tile_ready: cute.Tensor                    # [total_tiles] int64 — release stamps from dispatch's Pass 2
    tile_id_to_expert: cute.Tensor             # [total_tiles] int32 — per-tile expert lookup
    expert_pool_block_offset: cute.Tensor      # [E_local + 1] int32 — pool-block prefix-sum
    dispatch_seq: Int64
    total_tiles: Int32                         # passed as scalar so get_grid_shape doesn't deref device tensor


# ---------------------------------------------------------------------------
# Streaming kernel A class.
# ---------------------------------------------------------------------------
class StreamingMoeASm90(GemmGatedSm90):
    """Streaming-MoE kernel A: standard strided varlen_m GEMM + SwiGLU with
    queue-pull scheduler. Pool layout means kernel A uses the base GEMM
    mainloop's varlen_m path verbatim — no per-tile gather indirection.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bump the scheduler-to-consumer SMEM payload from 4 ints
        # (pid_m, pid_n, batch_idx, is_valid) to 5 ints (adds tile_id, used by
        # the postact destination override).
        self.sched_payload_ints = 5

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        trace_ptr: Optional[Int64] = None,
    ):
        """Type-shim override so CuTeDSL accepts StreamingTileSchedulerOptions
        as the scheduler_args type (base annotation is TileSchedulerOptions).
        Body delegates to GemmSm90.__call__ unchanged.
        """
        from quack.gemm_sm90 import GemmSm90 as _GemmSm90Base
        _GemmSm90Base.__call__(
            self, mA, mB, mD, mC, epilogue_args, scheduler_args, varlen_args, stream, trace_ptr,
        )

    # -- scheduler hooks -----------------------------------------------------

    def get_scheduler_class(self, varlen_m: bool = False):
        return StreamingTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,                # pool: (TK_padded, H)
        mB: cute.Tensor,                # W1: (2I, H, E_local)
        mD: Optional[cute.Tensor],      # None (no D for streaming kernel A)
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: VarlenArguments,
        epilogue_args,
    ):
        # mB shape is (n=2I, k=H, l=E_local); n-dim tile count = ceil(2I / tile_N).
        num_pid_n = cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1])
        E_local = cute.size(mB, mode=[2])
        return StreamingTileSchedulerArguments(
            problem_shape_ntile_mnl=(None, num_pid_n, E_local),
            consumer_head=scheduler_args.consumer_head,
            tile_ready=scheduler_args.tile_ready,
            tile_id_to_expert=scheduler_args.tile_id_to_expert,
            expert_pool_block_offset=scheduler_args.expert_pool_block_offset,
            dispatch_seq=scheduler_args.dispatch_seq,
            total_tiles=scheduler_args.total_tiles,
            tile_shape_mn=self.cta_tile_shape_mnk[:2],
            cluster_shape_mnk=self.cluster_shape_mnk,
            persistence_mode=PersistenceMode.STREAMING,
        )

    # -- postact destination override ----------------------------------------

    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Override: postact destination is postact_a[tile_id * tile_M : ..., :].

        Replaces the varlen path that uses cu_seqlens_m[batch_idx]. The mPostAct
        tensor is passed flat as (total_tiles * tile_M, I) so a row offset of
        tile_id * tile_M lands the per-tile slab.
        """
        import cutlass.utils.hopper_helpers as sm90_utils_og

        sPostAct = epi_smem_tensors[self._epi_smem_map["mPostAct"]]
        copy_atom_postact_r2s = sm90_utils_og.sm90_get_smem_store_op(
            self.postact_layout, self.postact_dtype, self.acc_dtype
        )
        tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_r2s)
        tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)

        tile_id = tile_coord_mnkl[2]
        row_offset = tile_id * self.cta_tile_shape_mnk[0]
        # The 2D postact tensor is wrapped as a ragged TMA tensor by
        # setup_epi_tensor (rank 3, ptr_shift=True). Use offset_ragged_tensor
        # to slice the per-tile slab.
        mPostAct_tile = copy_utils.offset_ragged_tensor(
            params.mPostAct,
            row_offset,
            self.cta_tile_shape_mnk[0],
            ragged_dim=0,
            ptr_shift=True,
        )
        copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
            params.tma_atom_mPostAct,
            mPostAct_tile,
            self.cta_tile_shape_postact_mn,
            params.epi_tile_mPostAct,
            sPostAct,
            tile_coord_mnkl,
        )
        return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact


# ---------------------------------------------------------------------------
# JIT compile factory.
# ---------------------------------------------------------------------------
@jit_cache
def _compile_streaming_moe_a(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    postact_dtype: Type[cutlass.Numeric],
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    activation: str,
    device_capacity,
):
    assert device_capacity[0] == 9, "Streaming MoE kernel A is SM90-only for now"
    assert activation in gate_fn_map, f"Need a gated activation; got {activation}"

    H_sym = cute.sym_int()
    I2_sym = cute.sym_int()
    I_sym = cute.sym_int()
    E_sym = cute.sym_int()
    TK_padded_sym = cute.sym_int()
    Mflat_sym = cute.sym_int()    # total_tiles * tile_m, in postact's M dim
    total_tiles_sym = cute.sym_int()
    cu_seqlens_len_sym = cute.sym_int()  # E_local + 1 at runtime

    # A: pool (TK_padded, H), k-major (H is contiguous).
    mA = fake_tensor(a_dtype, (TK_padded_sym, H_sym), leading_dim=1, divisibility=8)
    # B: W1 (2I, H, E_local), k-major per expert (H contiguous), batch dim = E_local.
    mB = fake_tensor(b_dtype, (I2_sym, H_sym, E_sym), leading_dim=1, divisibility=8)
    # No D output — streaming kernel A goes pool → SwiGLU → postact_a directly.
    mD = None
    mC = None
    # mPostAct: flat (total_tiles * tile_m, I), n-major (I contiguous).
    mPostAct = fake_tensor(postact_dtype, (Mflat_sym, I_sym), leading_dim=1, divisibility=8)

    # cu_seqlens_m drives the standard varlen_m m-offset for kernel A: each
    # entry is `expert_pool_block_offset[e] * tile_m`. Length E_local + 1.
    mCuSeqlensM = fake_tensor(cutlass.Int32, (cu_seqlens_len_sym,), leading_dim=0, divisibility=1)

    consumer_head = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)
    tile_ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    tile_id_to_expert = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    expert_pool_block_offset = fake_tensor(cutlass.Int32, (cu_seqlens_len_sym,), divisibility=1)

    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(0),  # set at runtime; 0 here keeps fake compile happy
        consumer_head=consumer_head,
        tile_ready=tile_ready,
        tile_id_to_expert=tile_id_to_expert,
        expert_pool_block_offset=expert_pool_block_offset,
        dispatch_seq=Int64(0),
        total_tiles=Int32(0),
    )

    epi_args = GemmGatedSm90.EpilogueArguments(
        mPostAct=mPostAct,
        act_fn=gate_fn_map[activation],
        rounding_mode=RoundingMode.RN,
    )

    varlen_args = VarlenArguments(
        mCuSeqlensM=mCuSeqlensM,
        mCuSeqlensK=None,
        mAIdx=None,
    )

    return compile_gemm_kernel(
        StreamingMoeASm90,
        a_dtype,
        (tile_m, tile_n),
        (cluster_m, cluster_n, 1),
        pingpong=False,
        persistent=True,
        gather_A=False,
        is_dynamic_persistent=False,
        device_capacity=device_capacity,
        mA=mA,
        mB=mB,
        mD=mD,
        mC=mC,
        epi_args=epi_args,
        scheduler_args=scheduler_args,
        varlen_args=varlen_args,
    )


# ---------------------------------------------------------------------------
# Test-only producer: walks tile_ready slot-by-slot and release-stores
# dispatch_seq on each, with a delay between fires. Used by tests to validate
# kernel A's per-tile spin without DeepEP.
# ---------------------------------------------------------------------------
class _StreamingTileProducer:
    @cute.jit
    def __call__(
        self,
        tile_ready: cute.Tensor,  # [total_tiles] int64
        total_tiles: cutlass.Int32,
        dispatch_seq: cutlass.Int64,
        delay_clocks: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(tile_ready, total_tiles, dispatch_seq, delay_clocks).launch(
            grid=[1, 1, 1], block=[1, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tile_ready: cute.Tensor,
        total_tiles: cutlass.Int32,
        dispatch_seq: cutlass.Int64,
        delay_clocks: cutlass.Int32,
    ):
        from quack import utils
        from cutlass._mlir.dialects import nvvm
        from cutlass.cutlass_dsl import T
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            for i in cutlass.range(total_tiles):
                start = cutlass.Int64(nvvm.read_ptx_sreg_clock64(T.i64()))
                end = start + cutlass.Int64(delay_clocks)
                while cutlass.Int64(nvvm.read_ptx_sreg_clock64(T.i64())) < end:
                    pass
                ready_ptr = utils.elem_pointer(tile_ready, (i,))
                utils.threadfence_system()
                utils.st_release_sys_global(ready_ptr, dispatch_seq)


@jit_cache
def _compile_streaming_tile_producer():
    total_tiles_sym = cute.sym_int()
    ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    op = _StreamingTileProducer()
    return cute.compile(
        op,
        ready, cutlass.Int32(0), cutlass.Int64(0), cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fire_tiles_with_delay(
    tile_ready: torch.Tensor,
    dispatch_seq: int,
    delay_us: int = 50,
) -> None:
    """Test helper: launches a single-thread producer kernel on the current
    CUDA stream that release-stores dispatch_seq into each slot of `tile_ready`
    with `delay_us` between fires.
    """
    assert tile_ready.dtype == torch.int64
    assert tile_ready.is_cuda and tile_ready.is_contiguous()
    total_tiles = tile_ready.shape[0]
    # H100 clock ~1.5 GHz → 1500 cycles/μs.
    delay_clocks = max(1, int(delay_us * 1500))
    compiled = _compile_streaming_tile_producer()
    compiled(
        tile_ready,
        cutlass.Int32(total_tiles),
        cutlass.Int64(dispatch_seq),
        cutlass.Int32(delay_clocks),
    )


def streaming_moe_a(
    pool: torch.Tensor,                       # (TK_padded, H) bf16 — k-major (pool data, expert-major)
    W1: torch.Tensor,                         # (E_local, 2I, H) bf16 — k-major per expert
    postact_a: torch.Tensor,                  # (total_tiles, tile_M, I) bf16
    tile_id_to_expert: torch.Tensor,          # (total_tiles,) int32
    expert_pool_block_offset: torch.Tensor,   # (E_local + 1,) int32 — pool-block prefix sum
    tile_ready: torch.Tensor,                 # (total_tiles,) int64 release stamps
    consumer_head: torch.Tensor,              # (1,) int32, caller resets to 0
    dispatch_seq: int,
    *,
    tile_m: int = 128,
    tile_n: int = 256,
    cluster_m: int = 1,
    cluster_n: int = 1,
    activation: str = "swiglu",
) -> None:
    """Launch streaming-MoE kernel A on the caller's current CUDA stream (pool layout).

    Caller is responsible for:
      - allocating postact_a as (total_tiles, tile_M, I) — kernel sees flat 2D.
      - resetting consumer_head to 0 before launch.
      - ensuring tile_ready is populated by the producer (DeepEP's
        Buffer.dispatch Pass 2 or a test stub) on a stream that release-stores
        tile_ready[tile_id] = dispatch_seq once the tile is ready.
      - launching this kernel on a stream different from the producer's so the
        per-tile spin actually waits rather than serializes.

    Numerical correctness: each CTA atomic-claims a linear work index, decomposes
    to (tile_id, pid_n), spins on tile_ready[tile_id], reads
    expert_id = tile_id_to_expert[tile_id] and pid_m = tile_id -
    expert_pool_block_offset[expert_id], strided-TMA-loads the pool[tile_id*tile_M : ...]
    rows (via the standard varlen_m m-offset), runs GEMM against W1[expert_id],
    applies SwiGLU, and TMA-stores into postact_a[tile_id, :, :].
    """
    assert pool.is_cuda and W1.is_cuda and postact_a.is_cuda
    assert pool.dim() == 2 and pool.is_contiguous()
    assert W1.dim() == 3
    assert postact_a.dim() == 3
    total_tiles, postact_tile_m, I = postact_a.shape
    assert postact_tile_m == tile_m
    assert tile_id_to_expert.shape == (total_tiles,)
    assert tile_ready.shape == (total_tiles,) and tile_ready.dtype == torch.int64
    assert consumer_head.shape == (1,) and consumer_head.dtype == torch.int32
    H = pool.shape[1]
    E_local = W1.shape[0]
    assert expert_pool_block_offset.shape == (E_local + 1,)
    assert W1.shape[1] == 2 * I, (
        f"W1 dim 1 must be 2*I = {2 * I}; got W1.shape={tuple(W1.shape)}"
    )
    assert W1.shape[2] == H, (
        f"W1 dim 2 (H) must match pool dim 1; got W1.shape={tuple(W1.shape)}, H={H}"
    )
    two_I = W1.shape[1]
    # Caller passes W1 as (E_local, 2I, H) k-major contiguous (each expert's
    # slab has H contiguous). We need the kernel to see shape (2I, H, E_local)
    # with leading_dim=1 (H is contiguous along K). torch.permute(1, 2, 0)
    # gives this layout WITHOUT a copy.
    W1_p = W1.permute(1, 2, 0)
    assert W1_p.stride(1) == 1, "W1[:,e,:] must be H-contiguous (caller passes k-major weights)"
    assert W1_p.shape == (two_I, H, E_local)

    # Flatten postact_a's leading two dims to (total_tiles * tile_m, I).
    postact_flat = postact_a.view(total_tiles * tile_m, I)

    # Build cu_seqlens_m = expert_pool_block_offset * tile_m. The standard
    # varlen_m path inside the GEMM uses this as the per-batch m-row offset:
    #   m_offset(tile) = cu_seqlens_m[batch_idx] + pid_m * tile_m
    #                  = expert_pool_block_offset[expert_id] * tile_m + tile_in_e * tile_m
    #                  = tile_id * tile_m
    # which lands at the correct pool row (pool is contiguous in tile_id order
    # by construction).
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * tile_m).contiguous()

    device_capacity = get_device_capacity(pool.device)
    assert device_capacity[0] == 9, "Streaming MoE kernel A is SM90-only for now"

    a_dtype = torch2cute_dtype_map[pool.dtype]
    b_dtype = torch2cute_dtype_map[W1.dtype]
    postact_dtype = torch2cute_dtype_map[postact_a.dtype]

    compiled_fn = _compile_streaming_moe_a(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        postact_dtype=postact_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        activation=activation,
        device_capacity=device_capacity,
    )

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_m * cluster_n)

    epi_args = GemmGatedSm90.EpilogueArguments(
        mPostAct=postact_flat,
        act_fn=None,  # Constexpr; pass None at call time
        rounding_mode=None,  # Constexpr; pass None at call time
    )
    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        consumer_head=consumer_head,
        tile_ready=tile_ready,
        tile_id_to_expert=tile_id_to_expert,
        expert_pool_block_offset=expert_pool_block_offset,
        dispatch_seq=Int64(dispatch_seq),
        total_tiles=Int32(total_tiles),
    )
    varlen_args = VarlenArguments(
        mCuSeqlensM=cu_seqlens_m,
        mCuSeqlensK=None,
        mAIdx=None,
    )

    compiled_fn(
        pool, W1_p, None, None,
        epi_args, scheduler_args, varlen_args, None,
    )
