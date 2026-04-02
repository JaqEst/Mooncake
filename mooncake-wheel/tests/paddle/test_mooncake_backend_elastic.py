import os
import time
import unittest
import multiprocessing as mp

import paddle
paddle.enable_compat(scope={"mooncake"})
import mooncake.distributed as dist
from mooncake import pg


broken_rank = 1


def _elastic_worker(rank, num_processes, signals):
    """Worker for testing elastic world size extension."""
    assert num_processes % 2 == 0

    if rank < num_processes // 2:
        # Ensure correct operation before extension
        world_size = num_processes // 2
        dist.init_process_group(
            backend="mooncake-cpu",
            rank=rank,
            world_size=world_size,
        )
        tensor = paddle.tensor([rank + 1], dtype=paddle.int32, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(1, world_size + 1))
        if rank == 0:
            signals["extend"] = 1

        # backend = dist.group.WORLD._get_backend(paddle.device("cpu"))
        # Extend world
        # Note: `extend_group_size_to` is non-blocking. Blocking will only
        # occur at the first communication if some peers have not yet connected.
        # This allows overlapping other operations between the group expansion
        # and the first communication call.
        dist.extend_group_size_to(num_processes)
    else:
        while "extend" not in signals:
            time.sleep(1)
        dist.init_process_group(
            backend="mooncake-cpu",
            rank=rank,
            world_size=num_processes,
        )

    # Ensure correct operation after extension
    tensor = paddle.tensor([rank + 1], dtype=paddle.int32, device="cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert tensor.item() == sum(range(1, num_processes + 1)), (
        f"Rank {rank} expected {sum(range(1, num_processes + 1))}, "
        f"get {tensor.item()}"
    )


def _recovery_worker(rank, num_processes, signals):
    """Worker for testing rank recovery."""
    if rank < num_processes:
        dist.init_process_group(
            backend="mooncake-cpu",
            rank=rank,
            world_size=num_processes,
        )
        if rank == broken_rank:
            return  # Simulate broken rank

        tensor = paddle.tensor([rank], dtype=paddle.int32, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(0, num_processes)) - broken_rank

        time.sleep(5)
        signals["recover"] = 1
        # backend = dist.group.WORLD._get_backend(paddle.device("cpu"))
        while True:
            (peer_state,) = dist.get_peer_state([broken_rank])
            if peer_state:
                break
        dist.recover_ranks([broken_rank])

        # Ensure correct operation after recovery
        tensor = paddle.tensor([rank], dtype=paddle.int32, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(0, num_processes)), (
            f"Rank {rank} expected {sum(range(0, num_processes))}, "
            f"get {tensor.item()}"
        )
    else:
        while "recover" not in signals:
            time.sleep(1)
        dist.init_process_group(
            backend="mooncake-cpu",
            rank=broken_rank,
            world_size=num_processes,
            pg_options=pg.MooncakeBackendOptions(
                paddle.ones((num_processes,), dtype=paddle.int32, device="cpu"),
                True,
            ),
        )

        # Ensure correct operation after recovery
        tensor = paddle.tensor([broken_rank], dtype=paddle.int32, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(0, num_processes)), (
            f"Rank {rank} expected {sum(range(0, num_processes))}, "
            f"get {tensor.item()}"
        )


class TestMooncakeBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "19000"

        mp.set_start_method('spawn')

    def test_elastic_extension(self):
        num_processes = 4
        mp_manager = mp.Manager()
        signals = mp_manager.dict()

        processes = []
        for rank in range(num_processes):
            p = mp.Process(
                target=_elastic_worker,
                args=(rank, num_processes, signals)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def test_rank_recovery(self):
        num_processes = 4
        mp_manager = mp.Manager()
        signals = mp_manager.dict()

        processes = []
        for rank in range(num_processes + 1):
            p = mp.Process(
                target=_recovery_worker,
                args=(rank, num_processes, signals)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == "__main__":
    unittest.main()
