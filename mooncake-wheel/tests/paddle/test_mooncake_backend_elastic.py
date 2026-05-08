import os
import time
import unittest
import multiprocessing as mp

import paddle
paddle.enable_compat(scope={"mooncake"})
import mooncake.distributed as dist
from mooncake import pg

USE_CUDA = os.getenv("MOONCAKE_TEST_USE_CUDA") == "1"
TEST_BACKEND = "mooncake" if USE_CUDA else "mooncake-cpu"
TEST_DEVICE = "cuda" if USE_CUDA else "cpu"

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "19000"

broken_rank = 1


def _set_device(rank):
    if USE_CUDA:
        paddle.cuda.set_device(rank)
        global TEST_DEVICE
        TEST_DEVICE = "cuda:{}".format(rank)


def _elastic_worker(rank, num_processes, signals):
    """Worker for testing elastic world size extension."""
    _set_device(rank)
    assert num_processes % 2 == 0
    if rank < num_processes // 2:
        # Ensure correct operation before extension
        world_size = num_processes // 2
        dist.init_process_group(
            backend=TEST_BACKEND,
            rank=rank,
            world_size=world_size,
            pg_options=pg.MooncakeBackendOptions(
                paddle.ones((world_size,), dtype=paddle.int32, device=TEST_DEVICE)
            ),
        )
        tensor = paddle.tensor([rank + 1], dtype=paddle.int32, device=TEST_DEVICE)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(1, world_size + 1))
        if rank == 0:
            signals["extend"] = 1

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
            backend=TEST_BACKEND,
            rank=rank,
            world_size=num_processes,
            pg_options=pg.MooncakeBackendOptions(
                paddle.ones((num_processes,), dtype=paddle.int32, device=TEST_DEVICE)
            ),
        )

    # Ensure correct operation after extension
    tensor = paddle.tensor([rank + 1], dtype=paddle.int32, device=TEST_DEVICE)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert tensor.item() == sum(range(1, num_processes + 1)), (
        f"Rank {rank} expected {sum(range(1, num_processes + 1))}, "
        f"get {tensor.item()}"
    )


def _deferred_recovery_worker(rank, num_processes, signals):
    """Worker for testing deferred rank recovery join."""
    if rank < num_processes:
        _set_device(rank)
        dist.init_process_group(
            backend=TEST_BACKEND,
            rank=rank,
            world_size=num_processes,
            pg_options=pg.MooncakeBackendOptions(
                paddle.ones((num_processes,), dtype=paddle.int32, device=TEST_DEVICE)
            ),
        )
        while dist.get_num_synced_ranks() < num_processes:
            time.sleep(0.1)
        if rank == broken_rank:
            return  # Simulate broken rank

        expected_without_broken = sum(range(0, num_processes)) - broken_rank
        tensor = paddle.tensor([rank], dtype=paddle.int32, device=TEST_DEVICE)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == expected_without_broken

        time.sleep(5)
        signals["recover"] = 1
        while True:
            (peer_state,) = dist.get_peer_state([broken_rank])
            if peer_state:
                break

        # Healthy ranks keep making progress before the recovered rank joins.
        tensor = paddle.tensor([rank], dtype=paddle.int32, device=TEST_DEVICE)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == expected_without_broken

        while "join_ready" not in signals:
            time.sleep(0.1)

        dist.recover_ranks([broken_rank])

        tensor = paddle.tensor([rank], dtype=paddle.int32, device=TEST_DEVICE)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(0, num_processes)), (
            f"Rank {rank} expected {sum(range(0, num_processes))}, "
            f"get {tensor.item()}"
        )
    else:
        _set_device(broken_rank)
        while "recover" not in signals:
            time.sleep(0.1)

        dist.init_process_group(
            backend=TEST_BACKEND,
            rank=broken_rank,
            world_size=num_processes,
            pg_options=pg.MooncakeBackendOptions(
                paddle.ones((num_processes,), dtype=paddle.int32, device=TEST_DEVICE),
                True,
            ),
        )

        # Deferred join starts in a local-only mode so collectives stay self-contained.
        tensor = paddle.tensor([broken_rank], dtype=paddle.int32, device=TEST_DEVICE)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == broken_rank

        signals["join_ready"] = 1
        dist.join_group()

        tensor = paddle.tensor([broken_rank], dtype=paddle.int32, device=TEST_DEVICE)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(0, num_processes)), (
            f"Rank {rank} expected {sum(range(0, num_processes))}, "
            f"get {tensor.item()}"
        )


class TestMooncakeBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

    def test_rank_recovery_deferred_join(self):
        num_processes = 4
        mp_manager = mp.Manager()
        signals = mp_manager.dict()

        processes = []
        for rank in range(num_processes + 1):
            p = mp.Process(
                target=_deferred_recovery_worker,
                args=(rank, num_processes, signals)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == "__main__":
    unittest.main()
