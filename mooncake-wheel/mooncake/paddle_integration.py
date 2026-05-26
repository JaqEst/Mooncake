from __future__ import annotations
from typing import List, Any

import paddle
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group
import paddle.distributed.collective as _col
from paddle.distributed.communication.group import destroy_process_group, _add_new_group

paddle.enable_compat(scope={"mooncake"})
from mooncake import ep, pg
import mooncake.distributed as dist


def _map_reduce_op(paddle_op):
    """
    Map ``paddle.framework.core.ReduceOp`` (C++ pybind enum) to Mooncake ReduceOp.

    Both enums share the same names (SUM/MAX/MIN/PRODUCT), so we resolve by
    name via getattr — one attribute lookup, no dict, no hash.
    """
    return getattr(pg.ReduceOp, paddle_op.name, pg.ReduceOp.SUM)


class MooncakePGAdapter:
    """
    Adapts ``mooncake.distributed.ProcessGroup`` to Paddle's C++ ProcessGroup
    duck-typing interface.

    Paddle's collective ops call methods on ``group.process_group`` with
    Paddle-style signatures. This class translates each call into the
    equivalent Mooncake backend call.

    Method mapping
    ──────────────────────────────────────────────────────────────────────────
    Paddle (called by paddle.distributed.*)    Mooncake (this class calls)
    ─────────────────────────────────────────  ──────────────────────────────
    all_reduce(t, op, sync_op)                 allreduce([t], AllreduceOptions)
    all_gather(out_list, t, sync_op)           allgather([out_list], [t], opts)
    all_gather_into_tensor(out, in, sync_op)   _allgather_base(out, in, opts)
    broadcast(t, src, sync_op)                 broadcast([t], BroadcastOptions)
    all_to_all(out_list, in_list, sync_op)     alltoall(out_list, in_list, opts)
    reduce_scatter*(out, in, op, sync_op)      _reduce_scatter_base(out, in, opts)
    barrier(device_id)                          barrier(BarrierOptions)
    send(t, dst, sync_op)                      send([t], dst)
    recv(t, src, sync_op)                      recv([t], src)
    name()                                     mc_group.get_backend_name()
    rank()                                     mc_group.rank()
    size()                                     mc_group.size()
    """

    def __init__(self, mc_group):
        self._mc = mc_group

    def name(self) -> str:
        return self._mc.get_backend_name()

    def rank(self) -> int:
        return self._mc.rank()

    def size(self) -> int:
        return self._mc.size()

    def all_reduce(self, tensor, op, sync_op: bool):
        # Paddle calls task.wait() itself when sync_op=True, so we don't wait here.
        opts = pg.AllreduceOptions()
        opts.reduce_op = _map_reduce_op(op)
        opts.async_op = not sync_op
        return self._mc.allreduce([tensor], opts)

    def all_reduce_on_calc_stream(self, tensor, op):
        # Paddle does NOT call task.wait() after on_calc_stream variants,
        # relying on stream ordering for synchronization. Mooncake operations
        # are inherently asynchronous, so we must wait explicitly here.
        opts = pg.AllreduceOptions()
        opts.reduce_op = _map_reduce_op(op)
        work = self._mc.allreduce([tensor], opts)
        work.wait()
        return work

    def all_gather(self, out_list: List, in_tensor, sync_op: bool):
        # Mooncake allgather uses double-wrapped lists: ([out_list], [in_tensor])
        opts = pg.AllgatherOptions()
        opts.async_op = not sync_op
        return self._mc.allgather([out_list], [in_tensor], opts)

    def all_gather_on_calc_stream(self, out_list: List, in_tensor):
        opts = pg.AllgatherOptions()
        work = self._mc.allgather([out_list], [in_tensor], opts)
        work.wait()
        return work

    def all_gather_into_tensor(self, out_tensor, in_tensor, sync_op: bool):
        opts = pg.AllgatherOptions()
        opts.async_op = not sync_op
        return self._mc._allgather_base(out_tensor, in_tensor, opts)

    def all_gather_into_tensor_on_calc_stream(self, out_tensor, in_tensor):
        opts = pg.AllgatherOptions()
        work = self._mc._allgather_base(out_tensor, in_tensor, opts)
        work.wait()
        return work

    def broadcast(self, tensor, src: int, sync_op: bool):
        opts = pg.BroadcastOptions()
        opts.root_rank = src
        opts.root_tensor = 0
        opts.async_op = not sync_op
        return self._mc.broadcast([tensor], opts)

    def broadcast_on_calc_stream(self, tensor, src: int):
        opts = pg.BroadcastOptions()
        opts.root_rank = src
        opts.root_tensor = 0
        work = self._mc.broadcast([tensor], opts)
        work.wait()
        return work

    def all_to_all(self, out_list: List, in_list: List, sync_op: bool):
        opts = pg.AllToAllOptions()
        opts.async_op = not sync_op
        return self._mc.alltoall(out_list, in_list, opts)

    def all_to_all_on_calc_stream(self, out_list: List, in_list: List):
        opts = pg.AllToAllOptions()
        work = self._mc.alltoall(out_list, in_list, opts)
        work.wait()
        return work

    def reduce_scatter(self, out_tensor, in_list, op, sync_op: bool):
        opts = pg.ReduceScatterOptions()
        opts.reduce_op = _map_reduce_op(op)
        opts.async_op = not sync_op
        return self._mc._reduce_scatter_base(out_tensor, in_list, opts)

    def reduce_scatter_on_calc_stream(self, out_tensor, in_list, op):
        opts = pg.ReduceScatterOptions()
        opts.reduce_op = _map_reduce_op(op)
        work = self._mc._reduce_scatter_base(out_tensor, in_list, opts)
        work.wait()
        return work

    def reduce_scatter_tensor(self, out_tensor, in_tensor, op, sync_op: bool):
        opts = pg.ReduceScatterOptions()
        opts.reduce_op = _map_reduce_op(op)
        opts.async_op = not sync_op
        return self._mc._reduce_scatter_base(out_tensor, in_tensor, opts)

    def reduce_scatter_tensor_on_calc_stream(self, out_tensor, in_tensor, op):
        opts = pg.ReduceScatterOptions()
        opts.reduce_op = _map_reduce_op(op)
        work = self._mc._reduce_scatter_base(out_tensor, in_tensor, opts)
        work.wait()
        return work

    def barrier(self, device_id: int = -1):
        # barrier is always synchronous; Paddle calls task.wait() after it too,
        # but a double-wait on a completed work is a no-op, so this is safe.
        opts = pg.BarrierOptions()
        work = self._mc.barrier(opts)
        work.wait()
        return work

    def send(self, tensor, dst: int, sync_op: bool):
        work = self._mc.send([tensor], dst)
        if sync_op:
            work.wait()
        return work

    def send_on_calc_stream(self, tensor, dst: int):
        work = self._mc.send([tensor], dst)
        work.wait()
        return work

    def recv(self, tensor, src: int, sync_op: bool):
        work = self._mc.recv([tensor], src)
        if sync_op:
            work.wait()
        return work

    def recv_on_calc_stream(self, tensor, src: int):
        work = self._mc.recv([tensor], src)
        work.wait()
        return work

    def _start_coalescing(self) -> None:
        pass

    def _end_coalescing(self, tasks=None) -> None:
        pass


def make_paddle_group(mc_group, group_id: int):
    return Group(
        rank_in_group=mc_group.rank(),
        id=group_id,
        ranks=mc_group.global_ranks,
        pg=MooncakePGAdapter(mc_group),
        name=mc_group.get_backend_name(),
    )


def new_mooncake_group(ranks: list, group_id: int):
    pg_opts = ep.MooncakeBackendOptions(
        paddle.zeros((len(ranks),), dtype=paddle.int32)
    )
    group = dist.new_group(
        ranks, backend="mooncake", pg_options=pg_opts
    )

    group = make_paddle_group(group, group_id=group_id)

    group_name = _col._default_group_name + str(group_id)
    _col._group_map_by_name[group_name] = group
    _col._group_map[group_id] = group
    _col._group_map_backend[group] = "mooncake"

    _add_new_group(group)

    return group


def init_mooncake_pg(
    world_size: int,
    rank: int,
    clean_existed_groups: bool = False,
    ib_device_filter: List = None,
    host_ip: str = None,
    logger: Any = None
) -> None:
    if host_ip:
        dist.set_host_ip(host_ip)

    if ib_device_filter:
        dist.set_device_filter(ib_device_filter)

    dist.init_process_group(
        backend="mooncake",
        store=_col._default_store,
        rank=rank,
        world_size=world_size,
        pg_options=pg.MooncakeBackendOptions(
            paddle.zeros((world_size,), dtype=paddle.int32)
        ),
    )

    if logger is not None:
        logger.info("[MooncakePG] init_mooncake_pg complete.")

    if clean_existed_groups is True:
        hcg = fleet.get_hybrid_communicate_group()
        for attr, val in list(vars(hcg).items()):
            if isinstance(val, Group):
                setattr(hcg, attr, None)

        gids = list(_col._group_map.keys())

        _col._group_map_backend.clear()
        _col._group_map.clear()
        _col._group_map_by_name.clear()

        destroy_process_group() # destory all existed groups

        world_group = make_paddle_group(dist._default_pg, group_id=0)

        _col._group_map_by_name[_col._default_group_name] = world_group
        _col._group_map[0] = world_group
        _col._group_map_backend[world_group] = "mooncake"

        _add_new_group(world_group)

        if logger is not None:
            logger.info(
                f"[MooncakePG] Removed NCCL group ids: {gids}. "
                f"Remaining group ids: {list(_col._group_map.keys())}"
            )
