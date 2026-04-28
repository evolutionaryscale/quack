# Copyright (c) 2025, Tri Dao.

from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Boolean, const_expr

import quack.utils as utils
from quack.fast_math import FastDivmod
from quack.pipeline import PipelineStateWAdvance
from quack.cute_dsl_utils import mlir_namedtuple


class RasterOrderOption(IntEnum):
    AlongM = 0
    AlongN = 1
    Heuristic = 2  # Pick AlongM if tiles_n > tiles_m, else AlongN


class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1


class PersistenceMode(IntEnum):
    NONE = 0
    STATIC = 1
    DYNAMIC = 2
    CLC = 3
    # STREAMING: persistent CTAs claim work via atomic_add on a dedicated
    # consumer_head, with each linear work index decomposing into
    # (streaming_tile_idx, pid_n). Each CTA acquire-spins on
    # tile_ready_queue_seq[streaming_tile_idx] until the producer (e.g.
    # DeepEP's streaming_slot_assign) releases the tile, then reads
    # tile_id = tile_ready_queue[streaming_tile_idx] and the per-tile
    # metadata (expert_id, recv_x_rows). Used by streaming-MoE kernel A.
    STREAMING = 4


@cute.jit
def get_raster_order_from_option(
    raster_order_option: RasterOrderOption, problem_shape_ncluster_mn: cute.Shape, group_size: Int32
) -> RasterOrder:
    raster_order = (
        RasterOrder.AlongM
        if raster_order_option == RasterOrderOption.AlongM
        else RasterOrder.AlongN
    )
    if raster_order_option == RasterOrderOption.Heuristic:
        problem_blocks_m = cute.round_up(problem_shape_ncluster_mn[0], group_size)
        problem_blocks_n = cute.round_up(problem_shape_ncluster_mn[1], group_size)
        raster_order = (
            RasterOrder.AlongM if problem_blocks_n > problem_blocks_m else RasterOrder.AlongN
        )
    return raster_order


# Grouping arguments together that should be passed to __call__
@mlir_namedtuple
class TileSchedulerOptions(NamedTuple):
    max_active_clusters: Int32
    raster_order: cutlass.Constexpr[RasterOrderOption] = RasterOrderOption.Heuristic
    max_swizzle_size: Int32 = Int32(8)
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None


@dataclass
class TileSchedulerArguments:
    problem_shape_ntile_mnl: cute.Shape
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.NONE


class TileScheduler:
    @dataclass
    class Params:
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_fdd: FastDivmod
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        num_clusters_in_group_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        batch_idx_permute: Optional[cute.Tensor]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "TileScheduler.Params":
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                cute.ceil_div(problem_shape_ntile_mn[0], args.cluster_shape_mnk[0]),
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)
            raster_order = get_raster_order_from_option(
                args.raster_order, problem_shape_ncluster_mn, args.group_size
            )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return TileScheduler.Params(
                problem_shape_ncluster_mnl,
                raster_order,
                FastDivmod(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod(group_size),
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(num_clusters_in_group),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.batch_idx_permute,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def _init_clc_mbarrier(sched_smem: Optional[cute.Tensor] = None, *, loc=None, ip=None) -> None:
        # We use 4 ints to store (pid_m, pid_n, batch_idx, is_valid),
        # another 4 ints to store clc response, and 2 ints to store the mbarrier for CLC
        # Since only the scheduler warp will touch the mbarrier (we don't use multicast when trying
        # to cancel workID), we only need the scheduler warp to initialize and sync.
        # If we use multicast when canceling workID, we would need all threads to sync.
        assert cute.size(sched_smem, mode=[0]) >= 12
        clc_mbar_ptr = sched_smem[None, 0].iterator + 8
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(clc_mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_warp()

    @staticmethod
    @cute.jit
    def _cluster_idx_to_work_idx_batch(
        params: Params, cluster_idx: Tuple[Int32, Int32, Int32], *, loc=None, ip=None
    ) -> Tuple[Int32, Optional[Int32]]:
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            current_work_idx = Int32(cluster_idx[0])
            batch_idx = Int32(cluster_idx[2])
            return current_work_idx, batch_idx
        else:
            current_work_idx = Int32(cluster_idx[2])
            batch_idx = None
            return current_work_idx, batch_idx

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TileScheduler":
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        current_work_idx, _ = TileScheduler._cluster_idx_to_work_idx_batch(
            params, cute.arch.cluster_idx(), loc=loc, ip=ip
        )
        stages = 0
        if const_expr(
            params.persistence_mode
            in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC, PersistenceMode.CLC]
        ):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        if const_expr(params.persistence_mode == PersistenceMode.CLC):
            if is_scheduler_warp:
                TileScheduler._init_clc_mbarrier(sched_smem, loc=loc, ip=ip)
        return TileScheduler(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return (
                params.cluster_shape_mnk[0] * cute.size(params.problem_shape_ncluster_mnl[:2]),
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * params.problem_shape_ncluster_mnl[2],
            )
        else:
            num_ctas_in_problem = cute.size(
                params.problem_shape_ncluster_mnl, loc=loc, ip=ip
            ) * cute.size(params.cluster_shape_mnk)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mnk, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # CTA Swizzle to promote L2 data reuse
        params = self.params
        group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
            # if cid_slow % 2 == 1:  # inner serpentine
            #     cid_fast_in_group = params.group_size_fdd.divisor - 1 - cid_fast_in_group
        else:  # tail part
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_tail_fdd)
            # if cid_slow % 2 == 1:  # inner serpentine
            #     cid_fast_in_group = params.group_size_tail_fdd.divisor - 1 - cid_fast_in_group
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.problem_shape_ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size_fdd.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def _cluster_id_to_cta_id(
        self, cid_m: Int32, cid_n: Int32, *, block_zero_only: bool = False, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        if const_expr(block_zero_only):
            bidx_in_cluster = (Int32(0), Int32(0))
        else:
            # Get the pid from cluster id
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.params.cluster_shape_mnk[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.params.cluster_shape_mnk[1] + bidx_in_cluster[1]
        return pid_m, pid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        params = self.params
        if const_expr(is_valid is None):
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            elif const_expr(params.persistence_mode == PersistenceMode.CLC):
                is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl[:2])
            else:
                is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl)
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
                cluster_id_in_problem = work_idx
                _, _, bidz_ = cute.arch.cluster_idx()
            else:
                bidz_, cluster_id_in_problem = divmod(work_idx, params.num_clusters_per_problem_fdd)
            if const_expr(bidz is not None):
                bidz_ = bidz
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = (
                bidz_
                if const_expr(params.batch_idx_permute is None)
                else params.batch_idx_permute[bidz_]
            )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        pid_m, pid_n, batch_idx, is_valid = Int32(0), Int32(0), Int32(0), Boolean(False)
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            pass
        # elif const_expr(params.persistence_mode == PersistenceMode.STATIC):
        #     return self._delinearize_work_idx(loc=loc, ip=ip)
        else:
            self._scheduler_pipeline.consumer_wait(self._pipeline_state)
            pid_m, pid_n, batch_idx, is_valid_i32 = [
                self._sched_smem[i, self._pipeline_state.index] for i in range(4)
            ]
            # Need this fence since the STAS from the producer is using the async proxy.
            # Without this, we get race condition / deadlock.
            if const_expr(cute.size(params.cluster_shape_mnk) > 1):
                cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                self._scheduler_pipeline.consumer_release(self._pipeline_state)
            self._pipeline_state.advance()
            is_valid = Boolean(is_valid_i32)
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, Boolean(is_valid))

    # @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        return self._delinearize_work_idx(self._current_work_idx, loc=loc, ip=ip)
        # if is_scheduler_warp:
        # work_tile_info = self._delinearize_work_idx(block_zero_only=True, loc=loc, ip=ip)
        # self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)
        # self.write_work_tile_to_smem(self._delinearize_work_idx(block_zero_only=True, loc=loc, ip=ip), loc=loc, ip=ip)

    @cute.jit
    def setup_initial_work_tile(
        self, is_scheduler_warp: bool | Boolean = False, *, loc=None, ip=None
    ) -> cutlass.utils.WorkTileInfo:
        """Hook for schedulers that need to fetch the first work tile from a
        producer (e.g. queue-pull) rather than decompose a static initial
        `_current_work_idx`. Default: just calls `initial_work_tile_info()`.
        Override in subclasses where the first tile must come from advance+get.
        """
        return self.initial_work_tile_info(loc=loc, ip=ip)

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None) -> Int32 | Tuple[Int32, Int32, Boolean]:
        """should only be called by the scheduler warp"""
        params = self.params
        num_persistent_clusters = cute.arch.cluster_dim()[2]
        if const_expr(params.persistence_mode == PersistenceMode.STATIC):
            return self._current_work_idx + num_persistent_clusters
            # Serpentine: alternate wave direction for a bit better load balancing
            # But currently seems a tiny bit slower, disabling for now.
            # c = Int32(cute.arch.cluster_idx()[2])
            # next_work_idx = self._current_work_idx + 2 * c + 1
            # if self.num_tiles_executed % 2 == 1:
            #     next_work_idx = self._current_work_idx + 2 * (num_persistent_clusters - 1 - c) + 1
            # return next_work_idx
        elif const_expr(params.persistence_mode == PersistenceMode.DYNAMIC):
            next_work_linear_idx = Int32(0)
            if cute.arch.lane_idx() == 0:
                # If varlen_m, problem_shape_ncluster_mnl[0] is None, so we use atomic_add
                # instead of atomic_inc, and at the end of the kernel must reset the semaphore to 0.
                #                 # cute.printf("before atomicadd, tidx = {}, bidz = {}, idx = {}", cute.arch.thread_idx()[0], cute.arch.block_idx()[2], current_work_idx)
                if const_expr(params.problem_shape_ncluster_mnl[0] is not None):
                    next_work_linear_idx = num_persistent_clusters + utils.atomic_inc_i32(
                        cute.size(params.problem_shape_ncluster_mnl) - 1,
                        params.tile_count_semaphore,
                    )
                else:  # varlen_m
                    next_work_linear_idx = num_persistent_clusters + utils.atomic_add_i32(
                        1, params.tile_count_semaphore
                    )
                # cute.printf("after atomicadd, tidx = {}, bidz = {}, idx = {}", cute.arch.thread_idx()[0], cute.arch.block_idx()[2], current_work_idx)
            return cute.arch.shuffle_sync(next_work_linear_idx, 0)
        elif const_expr(params.persistence_mode == PersistenceMode.CLC):
            clc_response_ptr = self._sched_smem[None, self._pipeline_state.index].iterator + 4
            mbarrier_addr = self._sched_smem[None, 0].iterator + 8
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbarrier_addr, 16, loc=loc, ip=ip)
                utils.issue_clc_query_nomulticast(mbarrier_addr, clc_response_ptr, loc=loc, ip=ip)
            cute.arch.sync_warp()
            cute.arch.mbarrier_wait(mbarrier_addr, self._pipeline_state.phase, loc=loc, ip=ip)
            bidx, bidy, bidz, valid = cute.arch.clc_response(clc_response_ptr, loc=loc, ip=ip)
            cute.arch.fence_view_async_shared()
            cluster_idx = (
                bidx // params.cluster_shape_mnk[0],
                bidy // params.cluster_shape_mnk[1],
                bidz // params.cluster_shape_mnk[2],
            )
            cluster_idx, batch_idx = type(self)._cluster_idx_to_work_idx_batch(
                params, cluster_idx, loc=loc, ip=ip
            )
            return cluster_idx, batch_idx, Boolean(valid)
        else:
            return Int32(0)

    @cute.jit
    def write_work_tile_to_smem(
        self, work_tile_info: cutlass.utils.WorkTileInfo, *, loc=None, ip=None
    ):
        params = self.params
        if const_expr(self._sched_smem is not None):
            # producer phase is always consumer_phase ^ 1
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_acquire(pipeline_state_producer)
            sched_data = [
                work_tile_info.tile_idx[0],
                work_tile_info.tile_idx[1],
                work_tile_info.tile_idx[3],
                Int32(work_tile_info.is_valid_tile),
            ]
            lane_idx = cute.arch.lane_idx()
            if lane_idx < cute.size(params.cluster_shape_mnk):
                # cute.printf("Producer pid_m = {}, pid_n = {}, batch_idx = {}, is_valid = {}, after empty wait, idx = {}", sched_data[0], sched_data[1], sched_data[2], sched_data[3], self._current_work_idx)
                pipeline_idx = self._pipeline_state.index
                if const_expr(cute.size(params.cluster_shape_mnk) == 1):
                    for i in cutlass.range_constexpr(4):
                        self._sched_smem[i, pipeline_idx] = sched_data[i]
                    self._scheduler_pipeline.producer_commit(self._pipeline_state)
                else:
                    peer_cta_rank_in_cluster = lane_idx
                    # Here we assume that the block idx in cluster is linearized such that
                    # x is the fastest moving direction, followed by y, then z.
                    bidx_in_cluster = peer_cta_rank_in_cluster % params.cluster_shape_mnk[0]
                    bidy_in_cluster = (
                        peer_cta_rank_in_cluster // params.cluster_shape_mnk[0]
                    ) % params.cluster_shape_mnk[1]
                    mbar_ptr = self._scheduler_pipeline.producer_get_barrier(self._pipeline_state)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, 16, peer_cta_rank_in_cluster)
                    utils.store_shared_remote_x4(
                        sched_data[0] + bidx_in_cluster,
                        sched_data[1] + bidy_in_cluster,
                        sched_data[2],
                        sched_data[3],
                        smem_ptr=self._sched_smem[None, pipeline_idx].iterator,
                        mbar_ptr=mbar_ptr,
                        peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                    )

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None,
        ip=None,
    ):
        """is_scheduler_warp should only be true for one warp in the whole cluster.
        Moreover, we assume that only block zero in the cluster is calling this function.
        If calling with is_scheduler_warp = True, advance_count must be 1.
        """
        params = self.params
        self.num_tiles_executed += Int32(advance_count)
        if const_expr(self._pipeline_state is not None and advance_count > 1):
            self._pipeline_state.advance_iters(advance_count - 1)
        if const_expr(params.persistence_mode in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC]):
            # We assume here that advance_count is 1 for scheduler_warp
            if is_scheduler_warp:
                self._current_work_idx = self._fetch_next_work_idx(loc=loc, ip=ip)
                work_tile_info = self._delinearize_work_idx(
                    self._current_work_idx, block_zero_only=True, loc=loc, ip=ip
                )
                self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)
        elif const_expr(params.persistence_mode == PersistenceMode.CLC):
            # We assume here that advance_count is 1 for scheduler_warp
            if is_scheduler_warp:
                self._current_work_idx, batch, is_valid = self._fetch_next_work_idx(loc=loc, ip=ip)
                work_tile_info = self._delinearize_work_idx(
                    self._current_work_idx, batch, is_valid, block_zero_only=True, loc=loc, ip=ip
                )
                self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)

    def producer_tail(self):
        if const_expr(self._scheduler_pipeline is not None):
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_tail(pipeline_state_producer)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self.num_tiles_executed,
            self._current_batch_idx,
            self._num_work_idx_before_cur_batch,
            self._sched_smem,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self.num_tiles_executed,
                self._current_batch_idx,
                self._num_work_idx_before_cur_batch,
                self._sched_smem,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


@cute.jit
def triangular_idx_to_coord(idx: Int32) -> Tuple[Int32, Int32]:
    """
    Convert a triangular index to 2D coordinates.
    This is used to convert the linear index to 2D coordinates for triangular matrices.
    """
    row = utils.ceil((utils.sqrt(2 * idx + 2.25) - 0.5)) - 1
    col = idx - (row * (row + 1)) // 2
    return row, col


class TriangularTileScheduler(TileScheduler):
    """We assume the tile size per cluster is square (e.g., 128 x 256 per CTA, with cluster 2 x 1)"""

    @dataclass
    class Params:
        problem_shape_ncluster_mnl: cute.Shape
        num_clusters_per_problem_fdd: FastDivmod
        group_size_inv_f32: Float32
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        group_size_mul_group_size_fdd: FastDivmod
        group_size_tail_mul_group_size_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "TriangularTileScheduler.Params":
            assert args.cluster_shape_mnk[2] == 1
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                cute.ceil_div(problem_shape_ntile_mn[0], args.cluster_shape_mnk[0]),
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            cluster_m = problem_shape_ncluster_mn[0]
            # Assume that each cluster is responsible for a square tile
            num_clusters_per_problem = cluster_m * (cluster_m + 1) // 2
            group_size = min(args.group_size, cluster_m)
            group_size_tail = cluster_m % group_size
            num_groups_regular = cluster_m // group_size
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return TriangularTileScheduler.Params(
                problem_shape_ncluster_mnl,
                FastDivmod(num_clusters_per_problem),
                Float32(1.0 / group_size),
                num_groups_regular,
                FastDivmod(group_size),
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(group_size * group_size),
                FastDivmod((group_size_tail if group_size_tail > 0 else 1) * group_size),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TriangularTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TriangularTileScheduler":
        current_work_idx, _ = TileScheduler._cluster_idx_to_work_idx_batch(
            params, cute.arch.cluster_idx(), loc=loc, ip=ip
        )
        stages = 0
        if const_expr(
            params.persistence_mode
            in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC, PersistenceMode.CLC]
        ):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        if const_expr(params.persistence_mode == PersistenceMode.CLC):
            if is_scheduler_warp:
                TileScheduler._init_clc_mbarrier(sched_smem, loc=loc, ip=ip)
        return TriangularTileScheduler(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        clusters = (params.num_clusters_per_problem_fdd.divisor, 1)
        num_ctas_mnl = (
            clusters[0] * params.cluster_shape_mnk[0],
            clusters[1] * params.cluster_shape_mnk[1],
            params.cluster_shape_mnk[2] * params.problem_shape_ncluster_mnl[2],
        )
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mnk, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # CTA Swizzle to promote L2 data reuse
        params = self.params
        group_size = params.group_size_fdd.divisor
        group_id = (
            utils.ceil(
                (utils.sqrt(2 * cluster_id_in_problem + 2.25) - 0.5) * params.group_size_inv_f32
            )
            - 1
        )
        cid_m_start = group_id * group_size
        id_in_group = cluster_id_in_problem - (cid_m_start * (cid_m_start + 1)) // 2
        group_size_actual = (
            group_size
            if group_id < params.num_groups_regular
            else params.group_size_tail_fdd.divisor
        )
        group_col, group_remainder = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            group_col, group_remainder = divmod(id_in_group, params.group_size_mul_group_size_fdd)
        else:  # tail part
            group_col, group_remainder = divmod(
                id_in_group, params.group_size_tail_mul_group_size_fdd
            )
        cid_m_in_group, cid_n_in_group = Int32(0), Int32(0)
        if id_in_group >= group_size_actual * group_size * group_id:  # triangular tail
            cid_m_in_group, cid_n_in_group = triangular_idx_to_coord(group_remainder)
        else:
            if group_id < params.num_groups_regular:
                cid_n_in_group, cid_m_in_group = divmod(group_remainder, params.group_size_fdd)
            else:
                cid_n_in_group, cid_m_in_group = divmod(group_remainder, params.group_size_tail_fdd)
        cid_m = cid_m_start + cid_m_in_group
        cid_n = group_col * group_size + cid_n_in_group
        return cid_m, cid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        params = self.params
        if const_expr(is_valid is None):
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            else:
                is_valid = (
                    work_idx
                    < params.num_clusters_per_problem_fdd.divisor
                    * params.problem_shape_ncluster_mnl[2]
                )
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
                cluster_id_in_problem = work_idx
                _, _, bidz_ = cute.arch.cluster_idx()
            else:
                bidz_, cluster_id_in_problem = divmod(work_idx, params.num_clusters_per_problem_fdd)
                cluster_id_in_problem = Int32(cluster_id_in_problem)  # divmod returns IntValue
            if const_expr(bidz is not None):
                bidz_ = bidz
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = bidz_
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        # tidx, _, _ = cute.arch.thread_idx()
        # if tidx == 0:
        #     cute.printf("bidx = {}, bidy = {}, group_id = {}, id_in_group = {}, group_size_actual = {}, group_col = {}, group_remainder = {}, cid_n_in_group = {}, cid_m_in_group = {}, cid_m = {}, cid_n = {}, is_valid = {}",
        #                 bidx, bidy, group_id, id_in_group, group_size_actual, group_col, group_remainder, cid_n_in_group, cid_m_in_group, cid_m, cid_n, is_valid)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)


@dataclass
class VarlenMTileSchedulerArguments:
    problem_shape_ntile_mnl: cute.Shape
    total_m: Int32
    cu_seqlens_m: cute.Tensor
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    tile_shape_mn: cutlass.Constexpr[cute.Shape]
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.NONE


class VarlenMTileScheduler(TileScheduler):
    @dataclass
    class Params:
        problem_shape_ncluster_mnl: cute.Shape
        total_m: Int32
        cu_seqlens_m: cute.Tensor
        raster_order: cutlass.Constexpr[RasterOrder]
        group_size: Int32
        group_size_fdd: Optional[FastDivmod]
        group_size_tail_fdd: Optional[FastDivmod]
        num_clusters_in_group_fdd: FastDivmod
        tile_shape_mn: cutlass.Constexpr[cute.Shape]
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "VarlenMTileScheduler.Params":
            # problem_shape_ntile_mnl[0] will be None for VarlenM
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                None,
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            raster_order = const_expr(
                RasterOrder.AlongM
                if args.raster_order == RasterOrderOption.AlongM
                else RasterOrder.AlongN  # For Heuristic we also use AlongN
            )
            ncluster_fast = problem_shape_ncluster_mn[
                0 if raster_order == RasterOrder.AlongM else 1
            ]
            ncluster_slow = problem_shape_ncluster_mn[
                1 if raster_order == RasterOrder.AlongM else 0
            ]
            if const_expr(ncluster_fast is not None):
                group_size = min(args.group_size, ncluster_fast)
                group_size_tail = ncluster_fast % group_size
            else:
                group_size, group_size_tail = args.group_size, None
            num_clusters_in_group = None
            if const_expr(ncluster_slow is not None):
                num_clusters_in_group = group_size * ncluster_slow
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return VarlenMTileScheduler.Params(
                problem_shape_ncluster_mnl,
                args.total_m,
                args.cu_seqlens_m,
                raster_order,
                group_size,
                FastDivmod(group_size) if ncluster_fast is not None else None,
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1)
                if group_size_tail is not None
                else None,
                FastDivmod(num_clusters_in_group) if num_clusters_in_group is not None else None,
                args.tile_shape_mn,
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return VarlenMTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def _cluster_idx_to_work_idx_batch(
        params: Params, cluster_idx: Tuple[Int32, Int32, Int32], *, loc=None, ip=None
    ) -> Tuple[Int32, Optional[Int32]]:
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            current_work_idx = Int32(cluster_idx[0])
        else:
            current_work_idx = Int32(cluster_idx[2])
        batch_idx = None
        return current_work_idx, batch_idx

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "VarlenMTileScheduler":
        current_work_idx, _ = VarlenMTileScheduler._cluster_idx_to_work_idx_batch(
            params, cute.arch.cluster_idx(), loc=loc, ip=ip
        )
        stages = 0
        if const_expr(
            params.persistence_mode
            in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC, PersistenceMode.CLC]
        ):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        if const_expr(params.persistence_mode == PersistenceMode.CLC):
            if is_scheduler_warp:
                TileScheduler._init_clc_mbarrier(sched_smem, loc=loc, ip=ip)
        return VarlenMTileScheduler(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        block_size = params.tile_shape_mn[0] * params.cluster_shape_mnk[0]
        num_batch = params.problem_shape_ncluster_mnl[2]
        total_clusters_m_max = (params.total_m + num_batch * (block_size - 1)) // block_size
        total_clusters_max = total_clusters_m_max * params.problem_shape_ncluster_mnl[1]
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return (
                params.cluster_shape_mnk[0] * total_clusters_max,
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2],
            )
        else:
            num_persistent_clusters = cutlass.min(max_active_clusters, total_clusters_max)
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, num_clusters_m: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        params = self.params
        # CTA Swizzle to promote L2 data reuse
        if const_expr(params.num_clusters_in_group_fdd is not None):
            group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
            num_clusters_in_group = params.num_clusters_in_group_fdd.divisor
        else:
            assert params.raster_order == RasterOrder.AlongN
            num_clusters_in_group = params.group_size * num_clusters_m
            group_id = cluster_id_in_problem // num_clusters_in_group
            id_in_group = cluster_id_in_problem - group_id * num_clusters_in_group
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if const_expr(params.group_size_fdd is not None and params.group_size_tail_fdd is not None):
            num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
            if (group_id + 1) * num_clusters_in_group <= num_clusters:
                cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
                # if cid_slow % 2 == 1:  # inner serpentine
                #     cid_fast_in_group = params.group_size_fdd.divisor - 1 - cid_fast_in_group
            else:  # tail part
                cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_tail_fdd)
                # if cid_slow % 2 == 1:  # inner serpentine
                #     cid_fast_in_group = params.group_size_tail_fdd.divisor - 1 - cid_fast_in_group
        else:
            assert params.raster_order == RasterOrder.AlongM
            group_size_actual = cutlass.min(
                params.group_size, num_clusters_m - group_id * params.group_size
            )
            cid_slow = id_in_group // group_size_actual
            cid_fast_in_group = id_in_group - cid_slow * group_size_actual
            # if cid_slow % 2 == 1:  # inner serpentine
            #     cid_fast_in_group = group_size_actual - 1 - cid_fast_in_group
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else num_clusters_m
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def _get_num_m_blocks(
        self, lane: Int32, bidb_start: Int32, block_size: cutlass.Constexpr[int]
    ) -> Int32:
        num_batch = self.params.problem_shape_ncluster_mnl[2]
        batch_idx = lane + bidb_start
        cur_cu_seqlen = Int32(0)
        if batch_idx <= num_batch:
            cur_cu_seqlen = self.params.cu_seqlens_m[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        return (
            cute.ceil_div(seqlen, block_size)
            if batch_idx < num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,  # not used
        is_valid_: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        assert bidz is None
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_batch = self.params.problem_shape_ncluster_mnl[2]
        block_size = params.tile_shape_mn[0] * params.cluster_shape_mnk[0]
        batch_idx = self._current_batch_idx
        next_tile_idx = work_idx

        problems_end_tile = self._num_work_idx_before_cur_batch
        num_clusters_m, num_clusters_cumulative, clusters_in_problems = Int32(0), Int32(0), Int32(0)
        is_valid = True
        if const_expr(is_valid_ is not None):
            is_valid = is_valid_
        if is_valid:
            while problems_end_tile <= next_tile_idx:
                num_clusters_m = self._get_num_m_blocks(
                    lane_idx, bidb_start=batch_idx, block_size=block_size
                )
                num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
                num_clusters_cumulative = utils.warp_prefix_sum(num_clusters, lane_idx)
                # Total number of blocks for the next 31 problems, same for all lanes
                clusters_in_problems = cute.arch.shuffle_sync(
                    num_clusters_cumulative, cute.arch.WARP_SIZE - 1
                )
                problems_end_tile += clusters_in_problems
                if problems_end_tile <= next_tile_idx:
                    batch_idx += cute.arch.WARP_SIZE - 1
                if batch_idx >= num_batch:
                    batch_idx = Int32(num_batch)
                    problems_end_tile = next_tile_idx + 1
        else:
            batch_idx = Int32(num_batch)

        is_valid = batch_idx < num_batch
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            is_valid &= self.num_tiles_executed == 0
        cid_m, cid_n = Int32(0), Int32(0)
        num_work_idx_before_cur_batch = self._num_work_idx_before_cur_batch
        if is_valid:
            problems_start_tile = problems_end_tile - clusters_in_problems
            # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, problems_end_tile = %d, num_clusters_m=%d, batch_idx = %d", self._tile_idx, problems_end_tile, num_clusters_m, batch_idx)
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_problems = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    problems_start_tile + num_clusters_cumulative <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_problems
            num_clusters_prev_lane = (
                0
                if batch_idx_in_problems == 0
                else cute.arch.shuffle_sync(num_clusters_cumulative, batch_idx_in_problems - 1)
            )
            num_clusters_m = cute.arch.shuffle_sync(num_clusters_m, batch_idx_in_problems)
            num_work_idx_before_cur_batch = problems_start_tile + num_clusters_prev_lane
            cluster_id_in_problem = next_tile_idx - num_work_idx_before_cur_batch
            # if cute.arch.thread_idx()[0] == 128: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, batch_idx=%d, cid_n=%d, cid_m=%d, is_valid = %d", self._tile_idx, batch_idx, cid_n, cid_m, is_valid)
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, num_clusters_m, loc=loc, ip=ip)
        pid_m, pid_n = self._cluster_id_to_cta_id(
            cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
        )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        self._current_batch_idx = batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)


class StreamingWorkTileInfo(cutlass.utils.WorkTileInfo):
    """WorkTileInfo variant whose tile_idx carries 4 ints (pid_m, pid_n, tile_id,
    batch_idx) rather than the standard 3 ints + None-K-slot. Used by
    StreamingTileScheduler so the per-tile gather + postact overrides can read
    tile_id directly from work_tile.tile_idx[2].
    """

    def __new_from_mlir_values__(self, values: list) -> "StreamingWorkTileInfo":
        assert len(values) == 5, f"StreamingWorkTileInfo expects 5 values, got {len(values)}"
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return StreamingWorkTileInfo(new_tile_idx, new_is_valid_tile)


@dataclass
class StreamingTileSchedulerArguments:
    """Arguments for the streaming-MoE tile scheduler. Produced by DeepEP's
    Buffer.dispatch and consumed by the QuACK streaming kernel.

    Linear-claim layout with per-tile ready signal:
      * tile_ready[total_tiles] int64 — slot_assign release-stores dispatch_seq
        into tile_ready[tile_id] when its tile_remaining hits zero. DeepEP's
        slot_assign walks experts in expert-major order, so tile_ready flips
        (becomes >= dispatch_seq) in expert-monotonic order.
      * consumer_head is a single [1] int32 — one global atomic-add counter.
        Linear claim order = tile_id order = expert-major order, so consumers
        naturally converge on the same expert at the same time. No window, no
        work-stealing, no per-CTA home expert.
    """

    problem_shape_ntile_mnl: cute.Shape  # (None, num_pid_n, num_local_experts)
    consumer_head: cute.Tensor           # [1] int32 — global linear claim counter
    tile_ready: cute.Tensor              # [total_tiles] int64 — release stamps from slot_assign
    tile_records_expert_id: cute.Tensor  # [total_tiles] int32 — per-tile expert lookup
    tile_records_recv_x_rows: cute.Tensor  # [total_tiles, tile_M] int32 — per-tile gather indices
    dispatch_seq: Int32                  # int64 in real use; kept Int32 here for kernel arg convenience
    total_tiles: Int32                   # passed as scalar so launch-time get_grid_shape doesn't deref device tensor
    tile_shape_mn: cutlass.Constexpr[cute.Shape]  # (tile_M, tile_N)
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.STREAMING


class StreamingTileScheduler(TileScheduler):
    """Linear-claim tile scheduler for streaming-MoE kernel A.

    Each persistent CTA's scheduler warp atomic-add-claims a linear work index
    `linear_idx = atomic_add(consumer_head, 1)`. The linear index decomposes
    into `(tile_id, pid_n) = divmod(linear_idx, num_pid_n)`. The scheduler
    spins on `tile_ready[tile_id]` until the producer releases
    (>= dispatch_seq), then reads `expert_id = tile_records_expert_id[tile_id]`.

    Wave behavior for free: DeepEP's slot_assign walks experts in expert-major
    order, firing tile_ready in tile_id order (since tile_id space is itself
    expert-grouped via cumulative_tiles_before_e). Linear claim order ==
    expert-major order, so 80 CTAs naturally converge on the same expert at the
    same time and L2 holds 1-2 W1[e] slabs throughout. No active-expert window
    or work-stealing logic in the scheduler.

    The work tile produced for the consumer warps carries:
      - `pid_m = 0` (each streaming tile is exactly tile_M rows; no per-tile M-tiling)
      - `pid_n` (the N-stripe)
      - `batch_idx = expert_id` (used by the existing kernel body to select W1[e])
      - `tile_id` (carried in the SMEM payload at the K-slot position)

    INTEGRATION NOTE: The existing `TileScheduler.write_work_tile_to_smem` writes
    4 ints (pid_m, pid_n, batch_idx, is_valid) to `_sched_smem` for the consumer
    warps. Streaming requires a 5th int — `tile_id` — so the consumer can use it
    for both A's gather index lookup (`tile_records_recv_x_rows[tile_id, :]`) and
    the postact_a destination (`postact_a[tile_id, :, pid_n × tile_N/2 : ...]`).
    The `_sched_smem` allocation site (in the kernel) needs to bump from 4 to 5
    ints; the `get_current_work` reader needs to unpack 5 ints; this class's
    `write_work_tile_to_smem` writes the 5th int.
    """

    @dataclass
    class Params:
        consumer_head: cute.Tensor
        tile_ready: cute.Tensor
        tile_records_expert_id: cute.Tensor
        tile_records_recv_x_rows: cute.Tensor
        dispatch_seq: Int32
        total_tiles: Int32
        num_pid_n: Int32
        num_pid_n_fdd: FastDivmod
        tile_shape_mn: cutlass.Constexpr[cute.Shape]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(
            args: StreamingTileSchedulerArguments, *, loc=None, ip=None
        ) -> "StreamingTileScheduler.Params":
            num_pid_n = cute.ceil_div(args.problem_shape_ntile_mnl[1], args.cluster_shape_mnk[1])
            return StreamingTileScheduler.Params(
                consumer_head=args.consumer_head,
                tile_ready=args.tile_ready,
                tile_records_expert_id=args.tile_records_expert_id,
                tile_records_recv_x_rows=args.tile_records_recv_x_rows,
                dispatch_seq=args.dispatch_seq,
                total_tiles=args.total_tiles,
                num_pid_n=num_pid_n,
                num_pid_n_fdd=FastDivmod(num_pid_n),
                tile_shape_mn=args.tile_shape_mn,
                cluster_shape_mnk=args.cluster_shape_mnk,
                persistence_mode=args.persistence_mode,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_tile_id: Int32,
        current_expert: Int32,
        current_pid_n: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        # Streaming scheduler state, persisted across the persistent loop's
        # iterations via the MLIR pytree round-trip:
        #   _current_tile_id: tile_id derived from the most recent linear claim.
        #     Used by _delinearize_work_idx and write_work_tile_to_smem so
        #     consumer warps see (pid_m, pid_n, tile_id, batch_idx) and can do
        #     per-tile gather + postact lookup.
        #   _current_expert: tile_records_expert_id[tile_id] for the current
        #     tile; surfaced via tile_coord_mnkl[3] for W1[e] selection.
        #   _current_pid_n: the N-stripe of the most recently claimed work,
        #     surfaced via tile_coord_mnkl[1].
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_tile_id = current_tile_id
        self._current_expert = current_expert
        self._current_pid_n = current_pid_n
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: StreamingTileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return StreamingTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "StreamingTileScheduler":
        # Initial work index: each persistent CTA starts unclaimed (work_idx = -1
        # means "not yet fetched"). The scheduler warp does its first
        # atomic_add(consumer_head) inside _fetch_next_work_idx during the first
        # advance_to_next_work call.
        stages = const_expr(cute.size(sched_smem, mode=[1])) if sched_smem is not None else 0
        return StreamingTileScheduler(
            current_work_idx=Int32(-1),
            num_tiles_executed=Int32(0),
            current_tile_id=Int32(-1),
            current_expert=Int32(0),
            current_pid_n=Int32(0),
            sched_smem=sched_smem,
            scheduler_pipeline=scheduler_pipeline,
            pipeline_state=PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params=params,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # Grid is sized to fill compute SMs. total_tiles is passed as a scalar
        # (not derived from cumulative_tiles_before_e[num_local_experts]) so
        # we don't need to dereference a device tensor at host launch time.
        total_work = params.total_tiles * params.num_pid_n
        num_persistent_clusters = cutlass.min(
            max_active_clusters, cute.ceil_div(total_work, cute.size(params.cluster_shape_mnk))
        )
        return (
            params.cluster_shape_mnk[0],
            params.cluster_shape_mnk[1],
            params.cluster_shape_mnk[2] * num_persistent_clusters,
        )

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None) -> Int32:
        """Scheduler-warp-only. Linear claim with per-tile ready spin.

        Lane 0 does ``linear_idx = atomic_add(consumer_head, 1)``. If
        ``linear_idx >= total_tiles * num_pid_n`` the kernel is exhausted and
        we return is_valid=0; consumer warps see is_valid_tile=False and exit
        the persistent loop. Otherwise:

          1. Decompose ``(tile_id, pid_n) = divmod(linear_idx, num_pid_n)``.
          2. Spin on ``tile_ready[tile_id]`` until value >= dispatch_seq —
             slot_assign release-stores it once tile_remaining[tile_id] hits 0.
          3. Read ``expert_id = tile_records_expert_id[tile_id]``.

        Because slot_assign processes tokens in expert-major order (per the
        pass-2 reorder in DeepEP), tile_ready flips in tile_id order and
        linear-claim CTAs naturally walk experts in waves: 80 CTAs all start
        on expert 0's tile range, drain it, advance to expert 1, etc.
        """
        params = self.params
        total_work = params.total_tiles * params.num_pid_n
        linear_idx = Int32(-1)
        if cute.arch.lane_idx() == 0:
            head_ptr = utils.elem_pointer(params.consumer_head, (Int32(0),))
            linear_idx = utils.atomic_add_i32(1, head_ptr)
        linear_idx = cute.arch.shuffle_sync(linear_idx, 0)

        is_valid_i32 = Int32(linear_idx < total_work)
        tile_id = Int32(-1)
        pid_n = Int32(0)
        expert_id = Int32(0)
        if is_valid_i32 != 0:
            tile_id, pid_n = divmod(linear_idx, params.num_pid_n_fdd)
            if cute.arch.lane_idx() == 0:
                ready_ptr = utils.elem_pointer(params.tile_ready, (tile_id,))
                while utils.ld_acquire_sys_global(ready_ptr) < cutlass.Int64(params.dispatch_seq):
                    pass
                expert_id = params.tile_records_expert_id[tile_id]
            expert_id = cute.arch.shuffle_sync(expert_id, 0)

        self._current_tile_id = tile_id
        self._current_expert = expert_id
        self._current_pid_n = pid_n
        return is_valid_i32

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        # _fetch_next_work_idx stashed the per-expert claim result onto self
        # and returned a 0/1 valid flag as the "linear work_idx" placeholder.
        # All work-tile components were determined inside _fetch; just thread
        # them through here.
        if const_expr(is_valid is None):
            is_valid = work_idx != Int32(0)
        pid_m = Int32(0)  # streaming tiles are always exactly tile_M rows
        # tile_coord_mnkl[2] (the K slot) carries tile_id for gather + postact;
        # tile_coord_mnkl[3] (batch_idx) is the expert_id we just claimed work
        # from, used by kernel body for W1[e] selection.
        tile_coord_mnkl = (
            pid_m,
            self._current_pid_n,
            self._current_tile_id,
            self._current_expert,
        )
        return StreamingWorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def write_work_tile_to_smem(
        self, work_tile_info: cutlass.utils.WorkTileInfo, *, loc=None, ip=None
    ):
        """Write 5 ints to _sched_smem: (pid_m, pid_n, tile_id, batch_idx=expert_id, is_valid).
        The 5th int (tile_id) is the streaming-specific extension; the consumer
        warps' get_current_work reader needs to unpack 5 ints in streaming mode.
        """
        params = self.params
        if const_expr(self._sched_smem is not None):
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_acquire(pipeline_state_producer)
            sched_data = [
                work_tile_info.tile_idx[0],  # pid_m
                work_tile_info.tile_idx[1],  # pid_n
                work_tile_info.tile_idx[2],  # tile_id (repurposed K slot)
                work_tile_info.tile_idx[3],  # batch_idx = expert_id
                Int32(work_tile_info.is_valid_tile),
            ]
            lane_idx = cute.arch.lane_idx()
            # Streaming uses cluster_shape_mnk = (1, 1, 1); multi-cluster would
            # need store_shared_remote_x5 (vs existing store_shared_remote_x4).
            assert cute.size(params.cluster_shape_mnk) == 1, (
                "StreamingTileScheduler currently requires cluster_shape == (1,1,1)"
            )
            if lane_idx < cute.size(params.cluster_shape_mnk):
                pipeline_idx = self._pipeline_state.index
                for i in cutlass.range_constexpr(5):
                    self._sched_smem[i, pipeline_idx] = sched_data[i]
                self._scheduler_pipeline.producer_commit(self._pipeline_state)

    @cute.jit
    def setup_initial_work_tile(
        self, is_scheduler_warp: bool | Boolean = False, *, loc=None, ip=None
    ) -> cutlass.utils.WorkTileInfo:
        """For streaming, the first work tile must come from the producer's
        atomic-claim + queue spin + sched_smem write — there is no static
        initial `_current_work_idx`. Both producer and consumer warps call
        this; producer's `advance_to_next_work` does the fetch+write, consumer's
        is a no-op; both then read the populated sched_smem via
        `get_current_work`.
        """
        self.advance_to_next_work(is_scheduler_warp=is_scheduler_warp, loc=loc, ip=ip)
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        """Streaming variant: unpacks 5 ints from sched_smem
        (pid_m, pid_n, tile_id, batch_idx=expert_id, is_valid). The tile_id is
        carried in tile_coord_mnkl[2] (the K slot — unused in GEMM context) so
        the consumer (gather setup, postact setup) can read it.
        """
        params = self.params
        self._scheduler_pipeline.consumer_wait(self._pipeline_state)
        pid_m, pid_n, tile_id, batch_idx, is_valid_i32 = [
            self._sched_smem[i, self._pipeline_state.index] for i in range(5)
        ]
        if const_expr(cute.size(params.cluster_shape_mnk) > 1):
            cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            self._scheduler_pipeline.consumer_release(self._pipeline_state)
        self._pipeline_state.advance()
        tile_coord_mnkl = (pid_m, pid_n, tile_id, batch_idx)
        return StreamingWorkTileInfo(tile_coord_mnkl, Boolean(is_valid_i32))

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None,
        ip=None,
    ):
        """Streaming variant. Same flow as TileScheduler.advance_to_next_work but
        always uses the STREAMING fetch path.
        """
        self.num_tiles_executed += Int32(advance_count)
        if const_expr(self._pipeline_state is not None and advance_count > 1):
            self._pipeline_state.advance_iters(advance_count - 1)
        if is_scheduler_warp:
            self._current_work_idx = self._fetch_next_work_idx(loc=loc, ip=ip)
            work_tile_info = self._delinearize_work_idx(
                self._current_work_idx, block_zero_only=True, loc=loc, ip=ip
            )
            self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self.num_tiles_executed,
            self._current_tile_id,
            self._current_expert,
            self._current_pid_n,
            self._sched_smem,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self.num_tiles_executed,
                self._current_tile_id,
                self._current_expert,
                self._current_pid_n,
                self._sched_smem,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)
