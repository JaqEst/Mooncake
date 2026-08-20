import os
from typing import Any, Callable, List, Tuple, Optional, Union

import paddle
import paddle.distributed as dist
paddle.enable_compat(scope={"mooncake"})
from mooncake import pg


def _map_reduce_op(paddle_op):
    """
    Map ``paddle.framework.core.ReduceOp`` (C++ pybind enum) to Mooncake ReduceOp.

    Both enums share the same names (SUM/MAX/MIN/PRODUCT), so we resolve by
    name via getattr — one attribute lookup, no dict, no hash.
    """
    return getattr(pg.ReduceOp, paddle_op.name, pg.ReduceOp.SUM)


def _mooncake_backend_options(
    world_size: int,
    *,
    device: str = "gpu",
    active_value: int = 0,
    is_extension: bool = False,
    max_world_size: int | None = None,
) -> pg.MooncakeBackendOptions:
    tensor_size = world_size if max_world_size is None else int(max_world_size)
    active_ranks = paddle.full(
        (tensor_size,),
        int(active_value),
        dtype=paddle.int32,
        device=device,
    )
    if max_world_size is None:
        return pg.MooncakeBackendOptions(active_ranks, is_extension)
    return pg.MooncakeBackendOptions(active_ranks, bool(is_extension), tensor_size)


class ProcessGroupMooncake:
    """
    Adapts MooncakeBackend to Paddle's C++ ProcessGroup interface via duck-typing.
    """
    def __init__(self, backend):
        self._backend = backend

    @staticmethod
    def create(
        store,
        rank: int,
        world_size: int,
        group_id: int,
        pg_options: dict,
        **kwargs,
    ) -> "ProcessGroupMooncake":
        prefix_store = pg.PrefixStore(str(group_id), store)

        # Mooncake backend options
        device = pg_options.get('device', "gpu")
        active_value = pg_options.get('active_value', 0)
        is_extension = pg_options.get('is_extension', False)
        max_world_size = pg_options.get('max_world_size', None)
        back_opts = _mooncake_backend_options(
            world_size,
            device=device,
            active_value=active_value,
            is_extension=is_extension,
            max_world_size=max_world_size
        )

        # Mooncake distributed backend options
        global_ranks_in_group = pg_options.get('global_ranks_in_group', None)
        assert global_ranks_in_group is not None, (
            "global_ranks_in_group must be provided for Mooncake backend"
        )
        dist_opts = pg.DistributedBackendOptions()
        dist_opts.store = prefix_store
        dist_opts.group_rank = rank
        dist_opts.group_size = world_size
        dist_opts.group_id = str(group_id)
        dist_opts.global_ranks_in_group = global_ranks_in_group

        if device == "gpu":
            backend = pg.createMooncakeBackend(dist_opts, back_opts)
        elif device == "cpu":
            backend = pg.createMooncakeCpuBackend(dist_opts, back_opts)
        else:
            raise ValueError(f"Unknown device: {device}")
        return ProcessGroupMooncake(backend)

    @staticmethod
    def set_host_ip(ip: str):
        pg.set_host_ip(ip)

    @staticmethod
    def set_device_filter(filters: List[str]):
        pg.set_device_filter(filters)

    @staticmethod
    def set_transfer_engine(engine):
        pg.set_transfer_engine(engine)

    def name(self) -> str:
        return self._backend.name()

    def rank(self) -> int:
        return self._backend.rank()

    def size(self) -> int:
        return self._backend.size()

    def recover_ranks(self, ranks: List[int]):
        pg.recover_ranks(self._backend, ranks)

    def join_group(self):
        pg.join_group(self._backend)

    def extend_group_size_to(self, new_size: int):
        pg.extend_group_size_to(self._backend, new_size)

    def get_num_synced_ranks(self):
        return pg.get_num_synced_ranks(self._backend)

    def get_active_ranks(self):
        return pg.get_active_ranks(self._backend)

    def get_peer_state(self, peer_ranks: List[int]):
        return pg.get_peer_state(self._backend, peer_ranks)

    def reduce(self, tensor, dst, op, sync_op: bool):
        # Paddle calls task.wait() itself when sync_op=True, so we don't wait here.
        opts = pg.ReduceOptions()
        opts.reduce_op = _map_reduce_op(op)
        opts.root_rank = dst
        opts.async_op = not sync_op
        return self._backend.reduce([tensor], opts)

    def reduce_on_calc_stream(self, tensor, dst, op):
        # Paddle does NOT call task.wait() after on_calc_stream variants,
        # relying on stream ordering for synchronization. Mooncake operations
        # are inherently asynchronous, so we must wait explicitly here.
        opts = pg.ReduceOptions()
        opts.reduce_op = _map_reduce_op(op)
        opts.root_rank = dst
        opts.async_op = False
        work = self._backend.reduce([tensor], opts)
        work.wait()
        return work

    def all_reduce(self, tensor, op, sync_op: bool):
        opts = pg.AllreduceOptions()
        opts.reduce_op = _map_reduce_op(op)
        opts.async_op = not sync_op
        return self._backend.allreduce([tensor], opts)

    def all_reduce_on_calc_stream(self, tensor, op):
        opts = pg.AllreduceOptions()
        opts.reduce_op = _map_reduce_op(op)
        work = self._backend.allreduce([tensor], opts)
        work.wait()
        return work

    def gather(self, in_tensor, out_list, dst, sync_op: bool, use_calc_stream: bool=False):
        opts = pg.GatherOptions()
        opts.root_rank = dst
        opts.async_op = not sync_op
        out = [out_list] if out_list else [[]]
        work = self._backend.gather(out, [in_tensor], opts)
        if use_calc_stream:
            work.wait()
        return work

    def all_gather(self, out_list: List, in_tensor, sync_op: bool):
        # Mooncake allgather uses double-wrapped lists: ([out_list], [in_tensor])
        opts = pg.AllgatherOptions()
        opts.async_op = not sync_op
        return self._backend.allgather([out_list], [in_tensor], opts)

    def all_gather_on_calc_stream(self, out_list: List, in_tensor):
        opts = pg.AllgatherOptions()
        work = self._backend.allgather([out_list], [in_tensor], opts)
        work.wait()
        return work

    def all_gather_into_tensor(self, out_tensor, in_tensor, sync_op: bool):
        opts = pg.AllgatherOptions()
        opts.async_op = not sync_op
        return self._backend._allgather_base(out_tensor, in_tensor, opts)

    def all_gather_into_tensor_on_calc_stream(self, out_tensor, in_tensor):
        opts = pg.AllgatherOptions()
        work = self._backend._allgather_base(out_tensor, in_tensor, opts)
        work.wait()
        return work

    def all_gather_partial(self, out_tensor, in_tensor, nranks, rank_id):
        raise NotImplementedError

    def all_gather_partial_on_calc_stream(self, out_tensor, in_tensor, nranks, rank_id):
        raise NotImplementedError

    def broadcast(self, tensor, src: int, sync_op: bool):
        opts = pg.BroadcastOptions()
        opts.root_rank = src
        opts.root_tensor = 0
        opts.async_op = not sync_op
        return self._backend.broadcast([tensor], opts)

    def broadcast_on_calc_stream(self, tensor, src: int):
        opts = pg.BroadcastOptions()
        opts.root_rank = src
        opts.root_tensor = 0
        work = self._backend.broadcast([tensor], opts)
        work.wait()
        return work

    def all_to_all(self, out_list: List, in_list: List, sync_op: bool):
        opts = pg.AllToAllOptions()
        opts.async_op = not sync_op
        return self._backend.alltoall(out_list, in_list, opts)

    def all_to_all_on_calc_stream(self, out_list: List, in_list: List):
        opts = pg.AllToAllOptions()
        work = self._backend.alltoall(out_list, in_list, opts)
        work.wait()
        return work

    def all_to_all_single(self, out_tensor, in_tensor, out_sizes, in_sizes, sync_op: bool):
        raise NotImplementedError

    def all_to_all_single_on_calc_stream(self, out_tensor, in_tensor, out_sizes, in_sizes):
        raise NotImplementedError

    def all_to_all_tensor(self, out_tensor, in_tensor, sync_op: bool):
        raise NotImplementedError

    def all_to_all_tensor_on_calc_stream(self, out_tensor, in_tensor):
        raise NotImplementedError

    def reduce_scatter(self, out_tensor, in_list, op, sync_op: bool):
        raise NotImplementedError

    def reduce_scatter_on_calc_stream(self, out_tensor, in_list, op):
        raise NotImplementedError

    def reduce_scatter_tensor(self, out_tensor, in_tensor, op, sync_op: bool):
        opts = pg.ReduceScatterOptions()
        opts.reduce_op = _map_reduce_op(op)
        opts.async_op = not sync_op
        return self._backend._reduce_scatter_base(out_tensor, in_tensor, opts)

    def reduce_scatter_tensor_on_calc_stream(self, out_tensor, in_tensor, op):
        opts = pg.ReduceScatterOptions()
        opts.reduce_op = _map_reduce_op(op)
        work = self._backend._reduce_scatter_base(out_tensor, in_tensor, opts)
        work.wait()
        return work

    def scatter(self, out_tensor, in_list, src, sync_op: bool):
        opts = pg.ScatterOptions()
        opts.root_rank = src
        opts.async_op = not sync_op
        inp = [in_list] if in_list else [[]]
        return self._backend.scatter([out_tensor], inp, opts)

    def scatter_on_calc_stream(self, out_tensor, in_list, src):
        opts = pg.ScatterOptions()
        opts.root_rank = src
        inp = [in_list] if in_list else [[]]
        work = self._backend.scatter([out_tensor], inp, opts)
        work.wait()
        return work

    def scatter_tensor(self, out_tensor, in_tensor, src, sync_op: bool):
        raise NotImplementedError

    def scatter_tensor_on_calc_stream(self, out_tensor, in_tensor, src):
        raise NotImplementedError

    def barrier(self, device_id: int = -1):
        # barrier is always synchronous; Paddle calls task.wait() after it too,
        # but a double-wait on a completed work is a no-op, so this is safe.
        opts = pg.BarrierOptions()
        work = self._backend.barrier(opts)
        work.wait()
        return work

    def send(self, tensor, dst: int, sync_op: bool):
        return self._backend.send([tensor], dst, 0)

    def send_on_calc_stream(self, tensor, dst: int):
        work = self._backend.send([tensor], dst, 0)
        work.wait()
        return work

    def send_partial(self, tensor, dst, nranks, rank_id, sync_op: bool=True):
        raise NotImplementedError

    def send_partial_on_calc_stream(self, tensor, dst, nranks, rank_id):
        raise NotImplementedError

    def recv(self, tensor, src: int, sync_op: bool):
        return self._backend.recv([tensor], src, 0)

    def recv_on_calc_stream(self, tensor, src: int):
        work = self._backend.recv([tensor], src, 0)
        work.wait()
        return work

    def recv_partial(self, tensor, src, nranks, rank_id, sync_op: bool=True):
        raise NotImplementedError

    def recv_partial_on_calc_stream(self, tensor, src, nranks, rank_id):
        raise NotImplementedError

    def _start_coalescing(self) -> None:
        pass

    def _end_coalescing(self, tasks=None) -> None:
        pass


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.upper() in {"1", "ON", "TRUE", "YES"}


_USE_MACA = (
    _env_enabled("MOONCAKE_EP_USE_MACA")
    or bool(getattr(paddle.version, "maca", None))
)
_USE_SPLIT_SEND_RECV = (
    _env_enabled("MOONCAKE_EP_USE_MUSA")
    or _USE_MACA
)


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(
        self,
        event: Optional["ep.EventHandle"] = None,
        extra_tensors: Optional[Tuple[paddle.Tensor, ...]] = None,
    ) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap():
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()


class Buffer:
    def __init__(self, group, num_ep_buffer_bytes: int = 0):
        from mooncake import ep

        # Initialize the CPP runtime
        self.group = group
        self.backend = self.group.process_group
        self.rank = self.backend.rank()
        self.group_size = self.backend.size()
        self.num_ep_buffer_bytes = num_ep_buffer_bytes
        # NIC auto-detection happens inside ep.Buffer via Topology::discover().
        self.runtime = ep.Buffer(
            self.rank, self.group_size, num_ep_buffer_bytes
        )
        # Fallback flag and buffers.
        # Note: `sync_nvlink_ipc_handles()` can mutate C++ `ibgda_disabled_` (True->False when
        # P2P+IPC succeeds for all ranks). We re-evaluate after IPC sync below.
        self._use_fallback = bool(self.runtime.ibgda_disabled())
        self._fallback_next_combine_buffer: Optional[paddle.Tensor] = None
        self._maca_phase_token: Optional[paddle.Tensor] = None
        self._maca_phase_recv_tokens: Optional[List[paddle.Tensor]] = None
        self._warned_active_ranks_without_mooncake_backend = False
        self.connect()

    def _maca_phase_fence(self, send_event: Optional[Any] = None) -> None:
        if not _USE_MACA:
            return

        backend = dist.get_backend(self.group)
        fence_device = paddle.device("cpu" if backend == "gloo" else "cuda")

        def wait_send_done() -> None:
            if send_event is not None:
                send_event.synchronize()
            else:
                paddle.cuda.synchronize()

        # Compatibility fence between SEND and RECV.  The EP payload still
        # uses the P2P fast path; this only keeps rank phases aligned on MACA.
        wait_send_done()
        if (
            self._maca_phase_token is None
            or self._maca_phase_token.device != fence_device
        ):
            self._maca_phase_token = paddle.empty(
                1, dtype=paddle.int32, device=fence_device
            )
        if (
            self._maca_phase_recv_tokens is None
            or self._maca_phase_recv_tokens[0].device != fence_device
        ):
            self._maca_phase_recv_tokens = [
                paddle.empty(1, dtype=paddle.int32, device=fence_device)
                for _ in range(self.group_size)
            ]
        self._maca_phase_token.fill_(1)
        ops = []
        for peer in range(self.group_size):
            if peer == self.rank:
                continue
            ops.append(
                dist.P2POp(
                    dist.isend, self._maca_phase_token, peer, self.group
                )
            )
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    self._maca_phase_recv_tokens[peer],
                    peer,
                    self.group,
                )
            )
        if not ops:
            return
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    def _wrap_maca_recv_hook(
        self, hook: Optional[Callable], send_event: Optional[Any]
    ) -> Callable:
        def wrapped_hook() -> None:
            self._maca_phase_fence(send_event)
            if hook is not None:
                hook()

        return wrapped_hook

    def connect(self, is_update: bool = False):
        from mooncake import ep

        if not self._use_fallback:
            (raddr, rkey) = self.runtime.get_mr_info()

            raddr = paddle.tensor([raddr], dtype=paddle.int64, device="cuda")
            raddrs = [
                paddle.empty(1, dtype=paddle.int64, device="cuda")
                for _ in range(self.group_size)
            ]
            dist.all_gather(raddrs, raddr, self.group)
            raddrs = paddle.cat(raddrs).tolist()

            rkey = paddle.tensor([rkey], dtype=paddle.int32, device="cuda")
            rkeys = [
                paddle.empty(1, dtype=paddle.int32, device="cuda")
                for _ in range(self.group_size)
            ]
            dist.all_gather(rkeys, rkey, self.group)
            rkeys = paddle.cat(rkeys).tolist()

            all_to_all_size = ep.MAX_QP_COUNT // self.group_size

            if is_update:
                self.runtime.update_local_qpns()

            local_qpns = self.runtime.get_local_qpns()
            local_qpns = list(
                paddle.unbind(
                    paddle.tensor(local_qpns, dtype=paddle.int32, device="cuda").view(
                        -1, all_to_all_size
                    )
                )
            )
            remote_qpns = [
                paddle.empty(all_to_all_size, dtype=paddle.int32, device="cuda")
                for _ in range(self.group_size)
            ]
            dist.alltoall(remote_qpns, local_qpns, self.group)
            peer_qpns = [remote_qpns[r].tolist() for r in range(self.group_size)]

            local_lids = self.runtime.get_local_lids()
            local_lids = list(
                paddle.unbind(
                    paddle.tensor(local_lids, dtype=paddle.int32, device="cuda").view(
                        -1, all_to_all_size
                    )
                )
            )
            remote_lids = [
                paddle.empty(all_to_all_size, dtype=paddle.int32, device="cuda")
                for _ in range(self.group_size)
            ]
            dist.alltoall(remote_lids, local_lids, self.group)
            peer_lids = [remote_lids[r].tolist() for r in range(self.group_size)]

            (subnet_prefix, interface_id) = self.runtime.get_gid()
            subnet_prefix_t = paddle.tensor([subnet_prefix], dtype=paddle.int64, device="cuda")
            subnet_prefixes_list = [
                paddle.empty(1, dtype=paddle.int64, device="cuda")
                for _ in range(self.group_size)
            ]
            dist.all_gather(subnet_prefixes_list, subnet_prefix_t, self.group)
            subnet_prefixes = paddle.cat(subnet_prefixes_list).tolist()

            interface_id_t = paddle.tensor([interface_id], dtype=paddle.int64, device="cuda")
            interface_ids_list = [
                paddle.empty(1, dtype=paddle.int64, device="cuda")
                for _ in range(self.group_size)
            ]
            dist.all_gather(interface_ids_list, interface_id_t, self.group)
            interface_ids = paddle.cat(interface_ids_list).tolist()

            active_ranks_mask = self._active_ranks_list(paddle.device("cuda"))
            self.runtime.sync_ibgda_peers(
                raddrs, rkeys, peer_qpns, peer_lids,
                subnet_prefixes, interface_ids, active_ranks_mask
            )

        if self.group_size == 1:
            # No peer can import this IPC handle in single-rank EP.  Skipping
            # export also avoids unnecessary driver IPC calls on MACA.
            self._use_fallback = False
            return
        else:
            try:
                local_handle_ints = self.runtime.get_ipc_handle()
                # pybind11 converts std::vector<int32_t> to a list of integers
                local_handle_tensor = paddle.tensor(
                    local_handle_ints, dtype=paddle.int32, device="cuda"
                )
                handles = [
                    paddle.empty(len(local_handle_ints), dtype=paddle.int32, device="cuda")
                    for _ in range(self.group_size)
                ]
                dist.all_gather(handles, local_handle_tensor, self.group)
                remote_handles = [h.tolist() for h in handles]
                active_ranks_mask = self._active_ranks_list(paddle.device("cuda"))
                self.runtime.sync_nvlink_ipc_handles(remote_handles, active_ranks_mask)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"[Rank {self.rank}] Failed to exchange IPC handles: {e}. Falling back.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        use_fast_path = False
        try:
            use_fast_path = bool(self.runtime.use_fast_path())
        except Exception:
            ibgda_disabled = bool(self.runtime.ibgda_disabled())
            use_fast_path = not ibgda_disabled

        self._use_fallback = not use_fast_path

    def update_ep_member(self):
        self.connect(True)

    def _is_mooncake_backend(self) -> bool:
        try:
            return dist.get_backend(self.group) == "mooncake"
        except Exception:
            return False

    def _active_ranks_tensor(
        self, device: paddle.device, dtype: paddle.dtype = paddle.int32
    ) -> paddle.Tensor:
        if not self._is_mooncake_backend():
            if not self._warned_active_ranks_without_mooncake_backend:
                import warnings

                try:
                    backend = dist.get_backend(self.group)
                except Exception:
                    backend = "unknown"
                warnings.warn(
                    "Mooncake EP active_ranks is only available with the "
                    f"mooncake process group; got backend={backend}. "
                    "Treating all ranks as active.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_active_ranks_without_mooncake_backend = True
            return paddle.ones((self.group_size,), dtype=dtype, device=device)

        try:
            return self.backend.get_active_ranks().to(device=device, dtype=dtype)
        except Exception:
            return paddle.ones((self.group_size,), dtype=dtype, device=device)

    def _active_ranks_list(self, device: paddle.device) -> List[int]:
        return self._active_ranks_tensor(device=device, dtype=paddle.int32).tolist()

    @staticmethod
    def get_ep_buffer_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        from mooncake.ep import get_ep_buffer_size_hint

        return get_ep_buffer_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

    # noinspection PyTypeChecker
    def dispatch(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        active_ranks: paddle.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        timeout_us: int,
        use_fp8: Optional[bool] = None,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[
        Union[Tuple[paddle.Tensor, paddle.Tensor], paddle.Tensor],
        paddle.Tensor,
        Tuple,
        EventOverlap,
        Callable,
    ]:
        if use_fp8 is None:
            use_fp8 = not _USE_MACA
        elif _USE_MACA and use_fp8:
            raise NotImplementedError("FP8 dispatch is not supported on MACA")

        # MUSA/MACA do not support cooperative grid sync, so the C++ runtime
        # splits no-hook calls into SEND -> phase-ack -> RECV instead of using
        # a single cooperative kernel.  async_finish still returns a stream
        # event, but it is not the CUDA single-kernel cooperative path.
        if _USE_SPLIT_SEND_RECV and async_finish:
            import warnings

            warnings.warn(
                "async_finish uses split SEND/RECV kernels plus a stream "
                "event, not CUDA cooperative single-kernel async semantics.",
                RuntimeWarning,
                stacklevel=2,
            )

        runtime_return_recv_hook = return_recv_hook or (
            _USE_MACA and not self._use_fallback
        )

        if self._use_fallback:
            (
                packed_recv_x,
                packed_recv_x_scales,
                packed_recv_count,
                packed_recv_src_info,
                packed_recv_layout_range,
                event,
                hook,
            ) = self._fallback_dispatch(
                x,
                topk_idx,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                use_fp8,
                return_recv_hook,
            )
            backend_active_ranks = self._active_ranks_tensor(
                device=active_ranks.device, dtype=active_ranks.dtype
            )
            if active_ranks.numel() == backend_active_ranks.numel():
                active_ranks.copy_(backend_active_ranks)
        else:
            (
                packed_recv_x,
                packed_recv_x_scales,
                packed_recv_count,
                packed_recv_src_info,
                packed_recv_layout_range,
                event,
                hook,
            ) = self.runtime.dispatch(
                x,
                topk_idx,
                active_ranks,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                timeout_us,
                use_fp8,
                async_finish,
                runtime_return_recv_hook,
            )
            if _USE_MACA:
                hook = self._wrap_maca_recv_hook(hook, event)
                if not return_recv_hook:
                    hook()
                    hook = None
        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
        )
        tensors_to_record = (
            x,
            topk_idx,
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )
        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    # noinspection PyTypeChecker
    def combine(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        active_ranks: paddle.Tensor,
        timeout_us: int,
        handle: tuple,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, EventOverlap, Callable]:
        # Same split-kernel behavior as dispatch().
        if _USE_SPLIT_SEND_RECV and async_finish:
            import warnings

            warnings.warn(
                "async_finish uses split SEND/RECV kernels plus a stream "
                "event, not CUDA cooperative single-kernel async semantics.",
                RuntimeWarning,
                stacklevel=2,
            )

        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        runtime_return_recv_hook = return_recv_hook or (
            _USE_MACA and not self._use_fallback
        )

        if self._use_fallback:
            combined_x, event, hook = self._fallback_combine(
                x,
                topk_idx,
                topk_weights,
                src_info,
                layout_range,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                zero_copy,
                return_recv_hook,
                out,
            )
            backend_active_ranks = self._active_ranks_tensor(
                device=active_ranks.device, dtype=active_ranks.dtype
            )
            if active_ranks.numel() == backend_active_ranks.numel():
                active_ranks.copy_(backend_active_ranks)
        else:
            combined_x, event, hook = self.runtime.combine(
                x,
                topk_idx,
                topk_weights,
                src_info,
                layout_range,
                active_ranks,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                timeout_us,
                zero_copy,
                async_finish,
                runtime_return_recv_hook,
                out,
            )
            if _USE_MACA:
                hook = self._wrap_maca_recv_hook(hook, event)
                if not return_recv_hook:
                    hook()
                    hook = None
        tensors_to_record = (
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combined_x,
        )
        return (
            combined_x,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    def get_next_combine_buffer(self, handle: object):
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        if self._use_fallback:
            if (
                self._fallback_next_combine_buffer is None
                or self._fallback_next_combine_buffer.shape
                != (
                    num_experts // self.group_size,
                    num_max_dispatch_tokens_per_rank * self.group_size,
                    hidden,
                )
            ):
                self._fallback_next_combine_buffer = paddle.empty(
                    (
                        num_experts // self.group_size,
                        num_max_dispatch_tokens_per_rank * self.group_size,
                        hidden,
                    ),
                    dtype=paddle.bfloat16,
                    device="cuda",
                )
            return self._fallback_next_combine_buffer
        return self.runtime.get_next_combine_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )

    # -----------------
    # Fallback helpers
    # -----------------
    class _DummyEvent:
        def current_stream_wait(self):
            paddle.cuda.synchronize()

    @staticmethod
    def _fp8_cast(x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        x_fp8 = (
            (x_view * (448.0 / x_amax.unsqueeze(2))).to(paddle.float8_e4m3fn).view(m, n)
        )
        x_scales = (x_amax / 448.0).view(m, -1)
        return x_fp8, x_scales

    def _fallback_dispatch(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        use_fp8: bool,
        return_recv_hook: bool,
    ):
        with paddle.profiler.RecordEvent("dispatch"):
            num_tokens, hidden = x.shape
            k = topk_idx.size(1)
            num_ranks = self.group_size
            num_local_experts = num_experts // num_ranks

            # Gather sizes first to handle variable num_tokens per rank
            num_tokens_tensor = paddle.tensor(
                [num_tokens], dtype=paddle.int64, device=x.device
            )
            num_tokens_list = [
                paddle.empty(1, dtype=paddle.int64, device=x.device)
                for _ in range(num_ranks)
            ]
            dist.all_gather(num_tokens_list, num_tokens_tensor, group=self.group)
            num_tokens_per_rank = [t.item() for t in num_tokens_list]
            backend_active_ranks = self._active_ranks_list(x.device)
            for i in range(num_ranks):
                if backend_active_ranks[i] == 0:
                    num_tokens_per_rank[i] = 0
            max_num_tokens = max(num_tokens_per_rank)

            # Pad inputs to max_num_tokens for all_gather (all ranks must have same shape)
            if num_tokens < max_num_tokens:
                pad_size = max_num_tokens - num_tokens
                x_padded = paddle.cat(
                    [
                        x,
                        paddle.zeros((pad_size, hidden), dtype=x.dtype, device=x.device),
                    ],
                    dim=0,
                )
                topk_padded = paddle.cat(
                    [
                        topk_idx,
                        paddle.full(
                            (pad_size, k), -1, dtype=topk_idx.dtype, device=x.device
                        ),
                    ],
                    dim=0,
                )
            else:
                x_padded = x
                topk_padded = topk_idx

            num_max_dispatch_tokens = num_ranks * num_max_dispatch_tokens_per_rank

            # Gather inputs from all ranks (all have same shape after padding)
            all_x = paddle.empty(
                (num_ranks, max_num_tokens, hidden), dtype=x.dtype, device=x.device
            )
            dist.stream.all_gather(all_x, x_padded, group=self.group)
            all_topk = paddle.empty(
                (num_ranks, max_num_tokens, k), dtype=topk_idx.dtype, device=x.device
            )
            dist.stream.all_gather(all_topk, topk_padded, group=self.group)

            # Prepare outputs per local expert
            recv_x_list: List[paddle.Tensor] = []
            recv_x_scales_list: List[paddle.Tensor] = []
            recv_count = paddle.zeros(
                (num_local_experts,), dtype=paddle.int32, device=x.device
            )
            recv_src_info = paddle.full(
                (num_local_experts, num_max_dispatch_tokens),
                -1,
                dtype=paddle.int32,
                device=x.device,
            )
            layout_range = paddle.zeros(
                (num_local_experts, num_ranks), dtype=paddle.int64, device=x.device
            )

            for le in range(num_local_experts):
                expert_id = self.rank * num_local_experts + le
                # Collect tokens from all ranks that route to this expert
                tokens_per_rank_tensors: List[paddle.Tensor] = []
                for src_rank in range(num_ranks):
                    src_num_tokens = num_tokens_per_rank[src_rank]
                    src_topk = all_topk[
                        src_rank, :src_num_tokens
                    ]  # Only consider valid tokens
                    # Find tokens that route to this expert
                    pos = (
                        (src_topk == expert_id)
                        .any(dim=1)
                        .nonzero(as_tuple=False)
                        .view(-1)
                    )
                    tokens_per_rank_tensors.append(pos)

                # Build ordered list grouped by src_rank (matching CUDA kernel behavior)
                begin = 0
                ordered_src_ranks_list: List[paddle.Tensor] = []
                ordered_token_indices_list: List[paddle.Tensor] = []
                for src_rank, tokens in enumerate(tokens_per_rank_tensors):
                    count = tokens.numel()
                    if count > 0:
                        layout_range[le, src_rank] = (begin << 32) | count
                        ordered_src_ranks_list.append(paddle.full_like(tokens, src_rank))
                        ordered_token_indices_list.append(tokens)
                        begin += count
                    else:
                        layout_range[le, src_rank] = 0

                if ordered_src_ranks_list:
                    ordered_src_ranks = paddle.cat(ordered_src_ranks_list)
                    ordered_token_indices = paddle.cat(ordered_token_indices_list)
                else:
                    ordered_src_ranks = paddle.empty(
                        0, dtype=topk_idx.dtype, device=x.device
                    )
                    ordered_token_indices = paddle.empty(
                        0, dtype=topk_idx.dtype, device=x.device
                    )

                num_valid = min(ordered_src_ranks.numel(), num_max_dispatch_tokens)
                recv_count[le] = num_valid

                # Materialize data
                if num_valid > 0:
                    gathered = all_x[
                        ordered_src_ranks[:num_valid], ordered_token_indices[:num_valid]
                    ]
                    src_meta = ordered_token_indices[:num_valid].to(dtype=paddle.int32)
                else:
                    gathered = paddle.empty(
                        (num_valid, hidden), dtype=paddle.bfloat16, device=x.device
                    )
                    src_meta = paddle.empty(
                        (num_valid,), dtype=paddle.int32, device=x.device
                    )

                # Pad to full size
                if use_fp8:
                    pad = num_max_dispatch_tokens - num_valid
                    if pad > 0:
                        pad_tensor = paddle.zeros(
                            (pad, hidden), dtype=paddle.bfloat16, device=x.device
                        )
                        gathered = paddle.cat([gathered, pad_tensor], dim=0)
                    fp8, scales = self._fp8_cast(gathered)
                    recv_x_list.append(fp8)
                    recv_x_scales_list.append(scales)
                else:
                    pad = num_max_dispatch_tokens - num_valid
                    if pad > 0:
                        pad_tensor = paddle.zeros(
                            (pad, hidden), dtype=paddle.bfloat16, device=x.device
                        )
                        gathered = paddle.cat([gathered, pad_tensor], dim=0)
                    recv_x_list.append(gathered)

                # src info padded
                if num_valid > 0:
                    recv_src_info[le, :num_valid] = src_meta

            if use_fp8:
                packed_recv_x = (
                    paddle.stack(recv_x_list, dim=0)
                    if len(recv_x_list) > 0
                    else paddle.empty(
                        (0, num_max_dispatch_tokens, hidden),
                        dtype=paddle.float8_e4m3fn,
                        device=x.device,
                    )
                )
                # Calculate scales shape correctly
                num_scales_per_token = hidden // 128
                packed_recv_x_scales = (
                    paddle.stack(recv_x_scales_list, dim=0)
                    if len(recv_x_scales_list) > 0
                    else paddle.empty(
                        (0, num_max_dispatch_tokens, num_scales_per_token),
                        dtype=paddle.float32,
                        device=x.device,
                    )
                )
            else:
                packed_recv_x = (
                    paddle.stack(recv_x_list, dim=0)
                    if len(recv_x_list) > 0
                    else paddle.empty(
                        (0, num_max_dispatch_tokens, hidden),
                        dtype=paddle.bfloat16,
                        device=x.device,
                    )
                )
                packed_recv_x_scales = None

            # Allocate zero-copy buffer for next combine
            self._fallback_next_combine_buffer = paddle.empty(
                (num_local_experts, num_max_dispatch_tokens, hidden),
                dtype=paddle.bfloat16,
                device=x.device,
            )

            hook = (lambda: None) if return_recv_hook else (lambda: None)
            event = Buffer._DummyEvent()
            return (
                packed_recv_x,
                packed_recv_x_scales,
                recv_count,
                recv_src_info,
                layout_range,
                event,
                hook,
            )

    def _fallback_combine(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        src_info: paddle.Tensor,
        layout_range: paddle.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        zero_copy: bool,
        return_recv_hook: bool,
        out: Optional[paddle.Tensor],
    ):
        with paddle.profiler.RecordEvent("combine"):
            num_tokens = topk_idx.size(0)
            hidden = (x if not zero_copy else self._fallback_next_combine_buffer).size(
                -1
            )
            num_ranks = self.group_size
            num_local_experts = num_experts // num_ranks

            # Gather sizes first to handle variable num_tokens per rank
            num_tokens_tensor = paddle.tensor(
                [num_tokens], dtype=paddle.int64, device=topk_idx.device
            )
            num_tokens_list = [
                paddle.empty(1, dtype=paddle.int64, device=topk_idx.device)
                for _ in range(num_ranks)
            ]
            dist.all_gather(num_tokens_list, num_tokens_tensor, group=self.group)
            num_tokens_per_rank = [t.item() for t in num_tokens_list]
            backend_active_ranks = self._active_ranks_list(topk_idx.device)
            for i in range(num_ranks):
                if backend_active_ranks[i] == 0:
                    num_tokens_per_rank[i] = 0
            max_num_tokens = max(num_tokens_per_rank)

            # Gather routing info across ranks to fetch per-token weights
            k = topk_idx.size(1)
            # Pad to max_num_tokens for all_gather
            if num_tokens < max_num_tokens:
                pad_size = max_num_tokens - num_tokens
                topk_padded = paddle.cat(
                    [
                        topk_idx,
                        paddle.full(
                            (pad_size, k),
                            -1,
                            dtype=topk_idx.dtype,
                            device=topk_idx.device,
                        ),
                    ],
                    dim=0,
                )
                topk_w_padded = paddle.cat(
                    [
                        topk_weights,
                        paddle.zeros(
                            (pad_size, k),
                            dtype=topk_weights.dtype,
                            device=topk_weights.device,
                        ),
                    ],
                    dim=0,
                )
            else:
                topk_padded = topk_idx
                topk_w_padded = topk_weights

            all_topk_idx = paddle.empty(
                (num_ranks, max_num_tokens, k),
                dtype=topk_idx.dtype,
                device=topk_idx.device,
            )
            dist.stream.all_gather(all_topk_idx, topk_padded, group=self.group)
            all_topk_w = paddle.empty(
                (num_ranks, max_num_tokens, k),
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
            dist.stream.all_gather(all_topk_w, topk_w_padded, group=self.group)

            expert_buffers = self._fallback_next_combine_buffer if zero_copy else x
            # Ensure bf16 input for accumulation
            if expert_buffers.dtype != paddle.bfloat16:
                # FP8 path should already have been cast back by caller before combine in tests
                expert_buffers = expert_buffers.to(paddle.bfloat16)

            # Build send buffer [num_ranks, max_num_tokens, hidden]
            send_buf = paddle.zeros(
                (num_ranks, max_num_tokens, hidden),
                dtype=paddle.bfloat16,
                device=expert_buffers.device,
            )

            for le in range(num_local_experts):
                expert_id = self.rank * num_local_experts + le
                # layout_range[le, j]: upper 32 begin, lower 32 count
                for src_rank in range(num_ranks):
                    entry = layout_range[le, src_rank]
                    begin = (entry >> 32).item() & 0xFFFFFFFF
                    count = entry.item() & 0xFFFFFFFF
                    if count == 0:
                        continue
                    tokens = src_info[le, begin : begin + count].to(paddle.long)
                    contrib = expert_buffers[le, begin : begin + count]

                    # Get source rank's actual token count and validate tokens
                    src_num_tokens = num_tokens_per_rank[src_rank]
                    valid_mask = tokens < src_num_tokens

                    if valid_mask.any():
                        tokens_valid = tokens[valid_mask]
                        contrib_valid = contrib[valid_mask]

                        # Find the per-token weight for this expert on src_rank
                        idx_rows = all_topk_idx[
                            src_rank, tokens_valid
                        ]  # [count_valid, k]
                        w_rows = all_topk_w[src_rank, tokens_valid]  # [count_valid, k]
                        mask = idx_rows == expert_id
                        weights = (w_rows * mask).sum(dim=1).view(-1, 1)
                        send_buf[src_rank, tokens_valid] += contrib_valid * weights

            # All-reduce then take local slice (only valid tokens)
            dist.all_reduce(send_buf, group=self.group)
            combined_x = send_buf[self.rank, :num_tokens]

            # Write to out if provided
            if out is not None:
                out.copy_(combined_x)
                combined = out
            else:
                combined = combined_x

            hook = (lambda: None) if return_recv_hook else (lambda: None)
            event = Buffer._DummyEvent()
            return combined, event, hook
