"""Streaming-MoE kernel A (CuTeDSL, SM90).

Forward kernel A of the problem-tile streaming pipeline (see new_design.md
"Pipeline architecture"):
  * Persistent CTAs pull tiles from a producer-fed queue.
  * Per-tile gather of recv_x rows via tile_records_recv_x_rows[tile_id, :].
  * GEMM against W1[expert_id] where expert_id = tile_records_expert_id[tile_id].
  * SwiGLU register-resident epilogue.
  * TMA-store the I-half post-activation to postact_a[tile_id * tile_M : ..., :].

Inherits the GEMM mainloop, SwiGLU epilogue, scheduler-warp + pipeline-state
machinery from quack.gemm_act.GemmGatedSm90. Streaming-specific behavior is
isolated to four overrides:
  (1) get_scheduler_class — return StreamingTileScheduler.
  (2) get_scheduler_arguments — build StreamingTileSchedulerArguments.
  (3) _make_gather_A_copy — gather indices come from tile_records_recv_x_rows[tile_id].
  (4) epi_setup_postact — postact destination indexed by tile_id, not by an
      mAIdx-derived varlen offset.

Plus the constructor sets `sched_payload_ints = 5` so the scheduler-to-consumer
SMEM payload carries tile_id alongside (pid_m, pid_n, batch_idx, is_valid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr

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
    consumer_head: cute.Tensor               # [1] int32 global linear claim counter
    tile_ready: cute.Tensor                  # [total_tiles] int64 — release stamps from slot_assign
    tile_records_expert_id: cute.Tensor      # [total_tiles] int32 — per-tile expert lookup
    dispatch_seq: Int64
    total_tiles: Int32                       # passed as scalar so get_grid_shape doesn't deref device tensor


# ---------------------------------------------------------------------------
# Streaming kernel A class.
# ---------------------------------------------------------------------------
class StreamingMoeASm90(GemmGatedSm90):
    """Streaming-MoE kernel A: gather + GEMM + SwiGLU with queue-pull scheduler.

    The base GemmGatedSm90 mainloop is reused verbatim. Only the four override
    points that depend on tile_id (rather than on a varlen `cu_seqlens_m` index)
    are specialized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bump the scheduler-to-consumer SMEM payload from 4 ints
        # (pid_m, pid_n, batch_idx, is_valid) to 5 ints (adds tile_id).
        # Read by GemmSm90.__call__ when allocating SharedStorage.sched_data
        # and when calling sched_data.get_tensor.
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
        mA: cute.Tensor,                # recv_x: (T_recv, H)
        mB: cute.Tensor,                # W1: (2I, H, E_local)
        mD: Optional[cute.Tensor],      # None (no D for streaming kernel A)
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: VarlenArguments,
        epilogue_args,
    ):
        # mB shape is (n=2I, k=H, l=E_local); n-dim tile count = ceil(2I / tile_N).
        num_pid_n = cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1])
        # E_local is taken from mB's L mode; the scheduler doesn't need it directly
        # under linear-claim, but problem_shape carries the shape for grid sizing.
        E_local = cute.size(mB, mode=[2])
        # varlen_args.mAIdx is tile_records_recv_x_rows of shape (total_tiles, tile_M).
        return StreamingTileSchedulerArguments(
            problem_shape_ntile_mnl=(None, num_pid_n, E_local),
            consumer_head=scheduler_args.consumer_head,
            tile_ready=scheduler_args.tile_ready,
            tile_records_expert_id=scheduler_args.tile_records_expert_id,
            tile_records_recv_x_rows=varlen_args.mAIdx,
            dispatch_seq=scheduler_args.dispatch_seq,
            total_tiles=scheduler_args.total_tiles,
            tile_shape_mn=self.cta_tile_shape_mnk[:2],
            cluster_shape_mnk=self.cluster_shape_mnk,
            persistence_mode=PersistenceMode.STREAMING,
        )

    # -- gather-A override ---------------------------------------------------

    @cute.jit
    def _make_gather_A_copy(
        self,
        mA_mkl: cute.Tensor,                # recv_x (T_recv, H) — flat 2D
        sA: cute.Tensor,
        varlen_manager,                     # carries mAIdx = tile_records_recv_x_rows
        tile_coord_mnkl,                    # (pid_m=0, pid_n, tile_id, expert_id)
        batch_idx: Int32,                   # = expert_id (unused on A side)
    ):
        """Per-tile gather: tile_M row indices from tile_records_recv_x_rows[tile_id, :]."""
        tile_id = tile_coord_mnkl[2]
        # tile_records_recv_x_rows shape (total_tiles, tile_M); slice tile_id row.
        gAIdx = varlen_manager.params.mAIdx[tile_id, None]
        mA_mk = mA_mkl  # full recv_x view, gather rows by gAIdx
        tiled_copy_A = self._make_gmem_tiled_copy_A(
            mA_mkl.element_type, self.a_layout, self.num_ab_load_warps * 32
        )
        dma_tidx = cute.arch.thread_idx()[0] - cute.arch.WARP_SIZE * self.ab_load_warp_id
        thr_copy_A = tiled_copy_A.get_slice(dma_tidx)
        # Each streaming tile is exactly tile_M rows by construction (sentinel
        # rows for partial tiles are written as -1 so the gather lands at row 0
        # which is harmless because the consumer mask drops them via
        # tile_records construction). limit_m = tile_M, limit_k = full H.
        limit_m = self.cta_tile_shape_mnk[0]
        limit_k = cute.size(mA_mkl, mode=[1])
        copy_A = copy_utils.gather_m_get_copy_fn(
            thr_copy_A, mA_mk, sA, gAIdx, limit_m=limit_m, limit_k=limit_k,
        )
        return copy_A, None  # no prefetch_A in m-major gather (varlen_m semantics)

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
        from functools import partial as _partial
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
    T_recv_sym = cute.sym_int()
    Mflat_sym = cute.sym_int()    # total_tiles * tile_m, in postact's M dim
    total_tiles_sym = cute.sym_int()

    # A: recv_x (T_recv, H), k-major (H is contiguous).
    mA = fake_tensor(a_dtype, (T_recv_sym, H_sym), leading_dim=1, divisibility=8)
    # B: W1 (2I, H, E_local), k-major per expert (H contiguous), batch dim = E_local.
    mB = fake_tensor(b_dtype, (I2_sym, H_sym, E_sym), leading_dim=1, divisibility=8)
    # No D output — streaming kernel A goes recv_x → SwiGLU → postact_a directly.
    mD = None
    mC = None
    # mPostAct: flat (total_tiles * tile_m, I), n-major (I contiguous).
    mPostAct = fake_tensor(postact_dtype, (Mflat_sym, I_sym), leading_dim=1, divisibility=8)

    # mAIdx is repurposed: (total_tiles, tile_M) per-tile gather indices.
    mAIdx = fake_tensor(cutlass.Int32, (total_tiles_sym, tile_m), leading_dim=1, divisibility=1)
    # Fake cu_seqlens_m so the base kernel's `gather_A => varlen_m or varlen_k`
    # assertion passes. Our override of _make_gather_A_copy ignores it; nothing
    # else in the streaming hot path consults cu_seqlens_m.
    mCuSeqlensM = fake_tensor(cutlass.Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)

    consumer_head = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)
    tile_ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    tile_records_expert_id = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)

    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(0),  # set at runtime; 0 here keeps fake compile happy
        consumer_head=consumer_head,
        tile_ready=tile_ready,
        tile_records_expert_id=tile_records_expert_id,
        dispatch_seq=Int64(0),
        total_tiles=Int32(0),
    )

    # mPostAct + act_fn (constexpr) + Nones for the unused alpha/beta/bias channels.
    epi_args = GemmGatedSm90.EpilogueArguments(
        mPostAct=mPostAct,
        act_fn=gate_fn_map[activation],
        rounding_mode=RoundingMode.RN,
    )

    varlen_args = VarlenArguments(
        mCuSeqlensM=mCuSeqlensM,
        mCuSeqlensK=None,
        mAIdx=mAIdx,
    )

    return compile_gemm_kernel(
        StreamingMoeASm90,
        a_dtype,
        (tile_m, tile_n),
        (cluster_m, cluster_n, 1),
        pingpong=False,
        persistent=True,
        gather_A=True,
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
# Host launch entry point.
# ---------------------------------------------------------------------------
class _StreamingTileProducer:
    """Test-only producer: walks tile_ready slot-by-slot and release-stores
    dispatch_seq on each, with a delay between fires. Used by
    test_streaming_producer_consumer to validate kernel A's per-tile spin.
    """

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


_CU_SEQLENS_M_SENTINEL: dict[torch.device, torch.Tensor] = {}


def _get_cu_seqlens_m_sentinel(device: torch.device) -> torch.Tensor:
    """Return a per-device int32 [2]-tensor used solely to satisfy the base
    kernel's `gather_A => varlen_m or varlen_k` assertion. The kernel body
    never dereferences it (our gather + postact + B-offset overrides bypass
    varlen_manager entirely), so the value is irrelevant; the only requirement
    is that it lives on the device and is non-None.

    Allocated lazily once per device and cached. Avoids a per-launch
    pageable-HtoD memcpy (and the implicit `cudaStreamSynchronize` that
    Pageable→Device staging triggers in PyTorch).
    """
    cached = _CU_SEQLENS_M_SENTINEL.get(device)
    if cached is None:
        cached = torch.zeros(2, dtype=torch.int32, device=device)
        _CU_SEQLENS_M_SENTINEL[device] = cached
    return cached


def streaming_moe_a(
    recv_x: torch.Tensor,                  # (T_recv, H) bf16 — k-major
    W1: torch.Tensor,                      # (E_local, 2I, H) bf16 — k-major per expert
    postact_a: torch.Tensor,               # (total_tiles, tile_M, I) bf16
    tile_records_recv_x_rows: torch.Tensor,  # (total_tiles, tile_M) int32
    tile_records_expert_id: torch.Tensor,    # (total_tiles,) int32
    tile_ready: torch.Tensor,                # (total_tiles,) int64 release stamps
    consumer_head: torch.Tensor,             # (1,) int32, caller resets to 0
    dispatch_seq: int,
    *,
    tile_m: int = 128,
    tile_n: int = 256,
    cluster_m: int = 1,
    cluster_n: int = 1,
    activation: str = "swiglu",
) -> None:
    """Launch streaming-MoE kernel A on the caller's current CUDA stream.

    Caller is responsible for:
      - allocating postact_a as (total_tiles, tile_M, I) — kernel sees flat 2D.
      - resetting consumer_head to 0 before launch.
      - ensuring tile_ready is populated by the producer (DeepEP's
        streaming_slot_assign or a test stub) on a stream that release-stores
        tile_ready[tile_id] = dispatch_seq once tile_remaining[tile_id] hits 0.
      - launching this kernel on a stream different from the producer's so the
        per-tile spin actually waits rather than serializes.

    Numerical correctness: each CTA atomic-claims a linear work index, decomposes
    to (tile_id, pid_n), spins on tile_ready[tile_id], gathers
    tile_records_recv_x_rows[tile_id, :] rows of recv_x, runs GEMM against
    W1[tile_records_expert_id[tile_id]], applies SwiGLU, and TMA-stores into
    postact_a[tile_id, :, :].
    """
    assert recv_x.is_cuda and W1.is_cuda and postact_a.is_cuda
    assert recv_x.dim() == 2 and recv_x.is_contiguous()
    assert W1.dim() == 3
    assert postact_a.dim() == 3
    total_tiles, postact_tile_m, I = postact_a.shape
    assert postact_tile_m == tile_m
    assert tile_records_recv_x_rows.shape == (total_tiles, tile_m)
    assert tile_records_expert_id.shape == (total_tiles,)
    assert tile_ready.shape == (total_tiles,) and tile_ready.dtype == torch.int64
    assert consumer_head.shape == (1,) and consumer_head.dtype == torch.int32
    H = recv_x.shape[1]
    E_local = W1.shape[0]
    # Caller passes W1 as (E_local, 2I, H) k-major contiguous (each expert's
    # slab has H contiguous). We need the kernel to see shape (2I, H, E_local)
    # with leading_dim=1 (H is contiguous along K). torch.permute(1, 2, 0)
    # on the (E_local, 2I, H) tensor gives this layout WITHOUT a copy:
    #   shape  (2I, H, E_local)
    #   stride (H, 1, 2I*H)        ← H stride = 1 (k-major per expert)
    assert W1.shape[1] == 2 * I, (
        f"W1 dim 1 must be 2*I = {2 * I}; got W1.shape={tuple(W1.shape)}"
    )
    assert W1.shape[2] == H, (
        f"W1 dim 2 (H) must match recv_x dim 1; got W1.shape={tuple(W1.shape)}, H={H}"
    )
    two_I = W1.shape[1]
    W1_p = W1.permute(1, 2, 0)
    assert W1_p.stride(1) == 1, "W1[:,e,:] must be H-contiguous (caller passes k-major weights)"
    assert W1_p.shape == (two_I, H, E_local)

    # Flatten postact_a's leading two dims to (total_tiles * tile_m, I).
    postact_flat = postact_a.view(total_tiles * tile_m, I)

    device_capacity = get_device_capacity(recv_x.device)
    assert device_capacity[0] == 9, "Streaming MoE kernel A is SM90-only for now"

    a_dtype = torch2cute_dtype_map[recv_x.dtype]
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
        tile_records_expert_id=tile_records_expert_id,
        dispatch_seq=Int64(dispatch_seq),
        total_tiles=Int32(total_tiles),
    )
    # Cached per-device int32[2] sentinel; satisfies base kernel's
    # `gather_A => varlen_m or varlen_k` assertion without a per-call
    # pageable HtoD memcpy + implicit stream sync.
    varlen_args = VarlenArguments(
        mCuSeqlensM=_get_cu_seqlens_m_sentinel(recv_x.device),
        mCuSeqlensK=None,
        mAIdx=tile_records_recv_x_rows,
    )

    compiled_fn(
        recv_x, W1_p, None, None,
        epi_args, scheduler_args, varlen_args, None,
    )
