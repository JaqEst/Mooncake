"""
Mooncake Distributed Communication Library
"""

import os
from datetime import timedelta
from typing import List, Optional

from mooncake import pg
from mooncake.pg import (
    ProcessGroup,
    PrefixStore,
    BroadcastOptions,
    AllreduceOptions,
    AllgatherOptions,
    ReduceScatterOptions,
    AllToAllOptions,
    BarrierOptions,
    ReduceOptions,
    ReduceOp,
    GatherOptions,
    ScatterOptions,
    DistributedBackendOptions,
)


# Special marker returned when current rank is not part of a sub-group
NON_GROUP_MEMBER = object()

# Global state
_default_pg: Optional[ProcessGroup] = None
_default_store = None
_pg_map = {}  # group_id -> ProcessGroup
_initialized = False


def init_process_group(
    backend: str = "mooncake",
    init_method: Optional[str] = None,
    store=None,
    rank: int = -1,
    world_size: int = -1,
    timeout: Optional[timedelta] = None,
    group_id: str = "default",
    pg_options=None,
):
    """
    Initialize the default distributed process group.

    Args:
        backend: Backend name, "mooncake" or "mooncake-cpu"
        init_method: URL string for rendezvous, e.g. "tcp://127.0.0.1:8361".
                     Only TCPStore is supported currently.
        store: Distributed store instance. Maybe Paddle.base.core.TCPStore, etc.
        rank: Rank of the current process
        world_size: Initial number of processes
        timeout: Timeout in seconds
        group_id: Process group ID
        pg_options: MooncakeBackendOptions instance provided by the user.
                    Create it via: pg_options = mooncake.pg.MooncakeBackendOptions(active_ranks_tensor)

    Raises:
        RuntimeError: If already initialized
        AssertionError: If both init_method and store are provided
        ValueError: If store is not provided, and init_method is not tcp://xxx:yyy
    """
    global _default_pg, _default_store, _initialized

    if _initialized:
        raise RuntimeError("Distributed already initialized")

    if not ((store is None) or (init_method is None)):
        raise AssertionError("Cannot specify both init_method and store.")

    if timeout is None:
        timeout = _get_default_timeout()

    # Get rank and world_size from environment if not provided
    if rank == -1:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if world_size == -1:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if store is None and init_method is None:
        init_method = (
            f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}"
            f":{os.getenv('MASTER_PORT', '8361')}"
        )

    # Try to get store from paddle if not provided
    if store is None:
        if init_method.startswith("tcp://"):
            addr = init_method[len("tcp://"):]
            host, port_str = addr.rsplit(":", 1)
            port = int(port_str)
            try:
                import paddle
                store = paddle.base.core.TCPStore(host, port, rank == 0, world_size)
            except Exception:
                store = None
        else:
            raise ValueError(
                f"Unsupported init_method scheme: '{init_method}'. "
                "Only 'tcp://' is currently supported."
            )

    if store is None:
        raise ValueError(
            "Please provide a compatible store instance (e.g., TCPStore from your framework). "
        )

    prefix_store = PrefixStore(group_id, store)
    _default_pg = _new_process_group_helper(
        backend, prefix_store, rank, world_size, timeout, group_id, pg_options,
        global_ranks=list(range(world_size))
    )
    _default_store = store
    _pg_map[group_id] = _default_pg
    _initialized = True


def new_group(
    ranks: List[int],
    backend: str = "mooncake",
    store=None,
    timeout: Optional[timedelta] = None,
    group_id: Optional[str] = None,
    pg_options=None,
):
    """
    Create a new process group with the specified ranks.

    This function creates a sub-process group containing only the specified ranks.
    All processes in the world are allowed to call this function, but only processes whose
    global rank is in `ranks` will get a valid group object returned.

    Args:
        ranks: List of global ranks to include in the new group (must be sorted)
        backend: Backend name ("mooncake" or "mooncake-cpu")
        store: Distributed store instance (if None, uses default store)
        timeout: Timeout in seconds
        group_id: Unique identifier for this group (auto-generated if None)
        pg_options: MooncakeBackendOptions instance for the new group.

    Returns:
        ProcessGroup object for the new group, or NON_GROUP_MEMBER if current rank is not in ranks

    Example:
        >>> # Create a group with ranks [0, 2] from a world of 4 processes
        >>> sub_group = dist.new_group(ranks=[0, 2])
        >>> if sub_group is not None:
        ...     # Only rank 0 and 2 will execute this
        ...     dist.all_reduce(tensor, group=sub_group)
    """
    _check_initialized()

    global_rank = _default_pg.rank()
    if global_rank not in ranks:
        # This process is not part of the new group
        _pg_map[group_id] = None
        return NON_GROUP_MEMBER

    if timeout is None:
        timeout = _get_default_timeout()

    # Compute local rank within the new group
    local_rank = ranks.index(global_rank)
    group_size = len(ranks)

    group_id = (group_id + "-" if group_id else "") + str(len(_pg_map))

    # Use the default store if not provided
    if store is None:
        store = _default_store

    prefix_store = PrefixStore(group_id, store)
    pg = _new_process_group_helper(
        backend, prefix_store, local_rank, group_size, timeout, group_id, pg_options,
        global_ranks=ranks
    )
    _pg_map[group_id] = pg
    return pg


def destroy_process_group(group=None):
    """
    Destroy a process group.

    Args:
        group: Process group to destroy. If None or default group, destroys all groups.
    """
    global _default_pg, _initialized, _pg_map

    if group is NON_GROUP_MEMBER:
        return

    if group is None or group is _default_pg:
        # Destroy all groups
        for pg in _pg_map.values():
            if pg is not None:
                pg.shutdown()
        _pg_map.clear()
        _default_pg = None
        _initialized = False
    else:
        group.shutdown()
        # Remove from map
        del _pg_map[group.group_id]


def is_initialized() -> bool:
    """Check if the distributed environment is initialized."""
    return _initialized


def get_default_group():
    """Get the default process group."""
    _check_initialized()
    return _default_pg


def get_group_by_id(group_id: str):
    """Get a process group by its ID.

    Args:
        group_id: The group identifier

    Returns:
        ProcessGroup object, or None if not found
    """
    _check_initialized()
    return _pg_map.get(group_id)


def get_backend(group=None) -> str:
    """Get the backend name of the process group."""
    group = _get_group(group)
    return group.name()


def get_rank(group=None) -> int:
    """Get the rank in the process group."""
    group = _get_group(group)
    return group.rank()


def get_group_rank(group=None) -> int:
    """Get the rank of the current process within the specified group.

    This is an alias for get_rank() for torch.distributed compatibility.
    """
    return get_rank(group)


def get_global_rank() -> int:
    """Get the global rank of the current process in the world."""
    _check_initialized()
    return _default_pg.rank()


def get_world_size(group=None) -> int:
    """Get the world size of the process group."""
    group = _get_group(group)
    return group.size()


def get_global_world_size() -> int:
    """Get the global world size (total number of processes)."""
    _check_initialized()
    return _default_pg.size()


def send(tensor, dst: int, group=None, tag: int = 0):
    """
    Send a tensor to the destination process (blocking).

    Args:
        tensor: Tensor to send
        dst: Destination rank
        group: Process group (default: default group)
        tag: Message tag
    """
    if _rank_not_in_group(group):
        return
    group = _get_group(group)
    work = group.send([tensor], dst, tag)
    work.wait()


def recv(tensor, src: int, group=None, tag: int = 0):
    """
    Receive a tensor from the source process (blocking).

    Args:
        tensor: Tensor to receive into
        src: Source rank
        group: Process group
        tag: Message tag
    """
    if _rank_not_in_group(group):
        return
    group = _get_group(group)
    work = group.recv([tensor], src, tag)
    work.wait()


def isend(tensor, dst: int, group=None, tag: int = 0):
    """
    Send a tensor asynchronously (non-blocking).

    Returns:
        Work object that can be waited on, or None if not part of the group.
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    return group.send([tensor], dst, tag)


def irecv(tensor, src: int, group=None, tag: int = 0):
    """
    Receive a tensor asynchronously (non-blocking).

    Returns:
        Work object that can be waited on, or None if not part of the group.
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    return group.recv([tensor], src, tag)


def batch_isend_irecv(p2p_op_list: List) -> List:
    """
    Batch multiple point-to-point operations.

    Args:
        p2p_op_list: List of P2POp objects (send/recv operations)

    Returns:
        List of Work objects
    """
    works = []
    for p2p_op in p2p_op_list:
        work = p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        works.append(work)
    return works


def broadcast(tensor, src: int = 0, group=None, async_op: bool = False):
    """
    Broadcast a tensor from the source to all processes.

    Args:
        tensor: Tensor to broadcast
        src: Source rank
        group: Process group
        async_op: If True, return Work object; otherwise block until complete

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = BroadcastOptions()
    opts.root_rank = src
    opts.root_tensor = 0
    opts.async_op = async_op
    work = group.broadcast([tensor], opts)

    if async_op:
        return work
    work.wait()
    return None


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op: bool = False):
    """
    Reduce tensor across all processes (in-place).

    Args:
        tensor: Input/output tensor
        op: Reduce operation ("sum", "avg", "min", "max", "product")
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = AllreduceOptions()
    opts.reduce_op = op
    opts.async_op = async_op
    work = group.allreduce([tensor], opts)

    if async_op:
        return work
    work.wait()
    return None


def all_gather(output_tensors: List, input_tensor, group=None, async_op: bool = False):
    """
    Gather tensors from all processes to all processes.

    Args:
        output_tensors: List of output tensors (length = world_size)
        input_tensor: Input tensor
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = AllgatherOptions()
    opts.async_op = async_op
    work = group.allgather([output_tensors], [input_tensor], opts)

    if async_op:
        return work
    work.wait()
    return None


def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op: bool = False):
    """
    Gather tensors from all processes into a single tensor.

    Args:
        output_tensor: Output tensor (size = input_tensor.size * world_size)
        input_tensor: Input tensor
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = AllgatherOptions()
    opts.async_op = async_op
    work = group._allgather_base(output_tensor, input_tensor, opts)

    if async_op:
        return work
    work.wait()
    return None


def reduce_scatter_tensor(output_tensor, input_tensor, op=ReduceOp.SUM, group=None, async_op: bool = False):
    """
    Reduce and scatter tensor across all processes.

    Args:
        output_tensor: Output tensor
        input_tensor: Input tensor
        op: Reduce operation
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = ReduceScatterOptions()
    opts.reduce_op = op
    opts.async_op = async_op
    work = group._reduce_scatter_base(output_tensor, input_tensor, opts)

    if async_op:
        return work
    work.wait()
    return None


def all_to_all(output_tensors: List, input_tensors: List, group=None, async_op: bool = False):
    """
    All-to-all communication.

    Args:
        output_tensors: List of output tensors
        input_tensors: List of input tensors
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = AllToAllOptions()
    opts.async_op = async_op
    work = group.alltoall(output_tensors, input_tensors, opts)

    if async_op:
        return work
    work.wait()
    return None


def reduce(tensor, dst: int = 0, op=ReduceOp.SUM, group=None, async_op: bool = False):
    """
    Reduce tensor to the destination process.

    Args:
        tensor: Input tensor
        dst: Destination rank
        op: Reduce operation
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = ReduceOptions()
    opts.reduce_op = op
    opts.root_rank = dst
    opts.async_op = async_op
    work = group.reduce([tensor], opts)

    if async_op:
        return work
    work.wait()
    return None


def gather(input_tensor, output_tensors: Optional[List] = None, dst: int = 0, group=None, async_op: bool = False):
    """
    Gather tensors to the destination process.

    Args:
        input_tensor: Input tensor
        output_tensors: List of output tensors (only needed on dst)
        dst: Destination rank
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = GatherOptions()
    opts.root_rank = dst
    opts.async_op = async_op
    out = [output_tensors] if output_tensors else [[]]
    work = group.gather(out, [input_tensor], opts)

    if async_op:
        return work
    work.wait()
    return None


def scatter(output_tensor, input_tensors: Optional[List] = None, src: int = 0, group=None, async_op: bool = False):
    """
    Scatter tensors from the source process.

    Args:
        output_tensor: Output tensor
        input_tensors: List of input tensors (only needed on src)
        src: Source rank
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = ScatterOptions()
    opts.root_rank = src
    opts.async_op = async_op
    inp = [input_tensors] if input_tensors else [[]]
    work = group.scatter([output_tensor], inp, opts)

    if async_op:
        return work
    work.wait()
    return None


def barrier(group=None, async_op: bool = False, device_ids = None, timeout = None):
    """
    Synchronization barrier across all processes.

    Args:
        group: Process group
        async_op: If True, return Work object; otherwise block

    Returns:
        Work object if async_op=True, otherwise None
    """
    if _rank_not_in_group(group):
        return None
    group = _get_group(group)
    opts = BarrierOptions()
    opts.async_op = async_op
    if timeout is not None:
        opts.timeout = timeout
    work = group.barrier(opts)

    if async_op:
        return work
    work.wait()
    return None


def set_host_ip(ip: str):
    """Set the host IP address."""
    pg.set_host_ip(ip)


def set_device_filter(filters: List[str]):
    """Set device filters."""
    pg.set_device_filter(filters)


def set_transfer_engine(engine):
    """Set the transfer engine."""
    pg.set_transfer_engine(engine)


def get_preferred_hca(location: str, group=None) -> str:
    """Get the preferred HCA for the given location."""
    group = _get_group(group)
    return pg.get_preferred_hca(group, location)


def get_active_ranks(group=None):
    """Get the tensor of active ranks."""
    group = _get_group(group)
    return pg.get_active_ranks(group)


def get_num_synced_ranks(group=None) -> int:
    """Get the number of synchronized ranks."""
    group = _get_group(group)
    return pg.get_num_synced_ranks(group)


def extend_group_size_to(new_size: int, group=None):
    """Extend the group size to the specified value."""
    group = _get_group(group)
    return pg.extend_group_size_to(group, new_size)


def get_peer_state(peer_ranks: List[int], group=None):
    """Get the state of peer ranks."""
    group = _get_group(group)
    return pg.get_peer_state(group, peer_ranks)


def recover_ranks(ranks: List[int], group=None):
    """Recover the specified ranks."""
    group = _get_group(group)
    return pg.recover_ranks(group, ranks)


def join_group(group=None):
    """Join the group."""
    group = _get_group(group)
    return pg.join_group(group)


def _check_initialized():
    if not _initialized:
        raise RuntimeError("Distributed not initialized. Call init_process_group() first.")


def _get_default_timeout():
    return timedelta(seconds=5 * 60)


def _get_group(group):
    """Get process group, using default if None."""
    _check_initialized()
    if group is None:
        return _default_pg
    return group


def _rank_not_in_group(group) -> bool:
    """
    Check if the current process's rank is not in the given group.

    Args:
        group: Process group to check

    Returns:
        True if current rank is NOT in the group, False otherwise
    """
    if group is NON_GROUP_MEMBER:
        return True
    # For None (default group) or actual process groups, we're in it
    return False


def _new_process_group_helper(backend, store, rank, world_size, timeout, group_id, pg_options, global_ranks=None):
    """Create a new ProcessGroup instance.

    Args:
        backend: Backend name ("mooncake" or "mooncake-cpu")
        store: Distributed store instance
        rank: Local rank within the group
        world_size: Size of the group
        timeout: Timeout in seconds
        group_id: Unique group identifier
        pg_options: MooncakeBackendOptions instance provided by the user
        global_ranks: List of global ranks in this group (for sub-groups)

    Returns:
        ProcessGroup instance
    """
    if global_ranks is None:
        global_ranks = list(range(world_size))

    dist_opts = DistributedBackendOptions()
    dist_opts.store = store
    dist_opts.group_rank = rank
    dist_opts.group_size = world_size
    dist_opts.group_id = group_id
    dist_opts.timeout = timeout.total_seconds()
    dist_opts.global_ranks_in_group = global_ranks

    if backend == "mooncake-cpu":
        group = pg.createMooncakeCpuBackend(dist_opts, pg_options)
    elif backend == "mooncake":
        group = pg.createMooncakeBackend(dist_opts, pg_options)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Supported backends are: 'mooncake', 'mooncake-cpu'"
        )

    return group


class P2POp:
    """Wrapper for point-to-point operations used with batch_isend_irecv."""

    def __init__(self, op: callable, tensor, peer: int, group=None, tag: int = 0):
        """
        Args:
            op: dist.isend or dis.irecv
            tensor: Tensor to send or receive
            peer: Destination rank (for send) or source rank (for recv)
            group: Process group
            tag: Message tag
        """
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag
