"""Drive the Mooncake process group through paddle.distributed.

Paddle treats a registered custom backend as a device backend, so
``init_parallel_env`` puts every rank on its own GPU: one visible GPU per rank.
"""

import os
from typing import List, Optional, Sequence

import paddle
paddle.enable_compat(scope={"mooncake"})
import paddle.distributed as dist
from mooncake.integration.paddle import ProcessGroupMooncake as pgmc

BACKEND = "mooncake"
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = 29500
DEFAULT_DEVICE_FILTERS = [f"mlx5_{i}" for i in range(1, 9)]


def device_count() -> int:
    return paddle.device.cuda.device_count()


def register_backend(device_filters: Optional[Sequence[str]] = None) -> None:
    """Register the Mooncake backend with paddle.distributed."""
    pgmc.set_device_filter(
        list(DEFAULT_DEVICE_FILTERS if device_filters is None else device_filters)
    )
    dist.register_process_group_backend(BACKEND, pgmc.create)


def mooncake_pg_options(
    global_ranks: List[int],
    *,
    active_value: int = 0,
    is_extension: bool = False,
    max_world_size: Optional[int] = None,
) -> dict:
    """Build the pg_options dict consumed by ProcessGroupMooncake.create."""
    return {
        "active_value": active_value,
        "is_extension": is_extension,
        "max_world_size": max_world_size,
        "global_ranks_in_group": list(global_ranks),
    }


def init_world_group(
    rank: int,
    world_size: int,
    *,
    device_id: Optional[int] = None,
    active_value: int = 0,
    is_extension: bool = False,
    max_world_size: Optional[int] = None,
):
    """Bring up the Mooncake-backed WORLD group.

    ``world_size`` is what this rank believes the group size to be, so elastic
    tests can have survivors and joiners disagree on it.
    """
    device_id = rank if device_id is None else device_id
    paddle.set_device(f"gpu:{device_id}")
    register_backend()

    master_addr = os.environ.get("MASTER_ADDR", DEFAULT_MASTER_ADDR)
    master_port = int(os.environ.get("MASTER_PORT", DEFAULT_MASTER_PORT))
    endpoints = ",".join(f"{master_addr}:{master_port + r}" for r in range(world_size))
    os.environ.update(
        {
            "PADDLE_DISTRI_BACKEND": BACKEND,
            "FLAGS_selected_gpus": str(device_id),
            "PADDLE_TRAINER_ID": str(rank),
            "PADDLE_TRAINERS_NUM": str(world_size),
            "PADDLE_TRAINER_ENDPOINTS": endpoints,
            "PADDLE_CURRENT_ENDPOINT": f"{master_addr}:{master_port + rank}",
        }
    )

    return dist.init_parallel_env(
        pg_options=mooncake_pg_options(
            list(range(world_size)),
            active_value=active_value,
            is_extension=is_extension,
            max_world_size=max_world_size,
        )
    )


def new_mooncake_group(
    ranks: List[int],
    *,
    active_value: int = 1,
    is_extension: bool = False,
    max_world_size: Optional[int] = None,
    gid: Optional[int] = None,
):
    """Create a sub-group over ``ranks``.

    Pin ``gid`` explicitly: Paddle derives the store prefix from a counter that
    only advances on ranks which are members of the group.
    """
    if gid is not None:
        dist.collective._set_custom_gid(gid)
    try:
        return dist.new_group(
            ranks,
            backend=BACKEND,
            pg_options=mooncake_pg_options(
                ranks,
                active_value=active_value,
                is_extension=is_extension,
                max_world_size=max_world_size,
            ),
        )
    finally:
        if gid is not None:
            dist.collective._set_custom_gid(None)
