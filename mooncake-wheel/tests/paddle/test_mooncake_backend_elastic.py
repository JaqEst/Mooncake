import time
import unittest
import multiprocessing as mp

import paddle
import paddle.distributed as dist

from pg_test_utils import device_count, init_world_group, new_mooncake_group

broken_rank = 1


def _elastic_worker(rank, num_processes, signals):
    """Worker for testing elastic world size extension."""
    assert num_processes % 2 == 0

    world_size = num_processes // 2
    extend_ranks = list(range(world_size, num_processes))

    if rank < world_size:
        # Ensure correct operation before extension
        world = init_world_group(rank, world_size, max_world_size=num_processes)
        backend = world.process_group

        tensor = paddle.to_tensor([rank + 1], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(1, world_size + 1))

        if rank == 0:
            signals["extend"] = 1
        while not all(backend.get_peer_state(extend_ranks)):
            time.sleep(0.1)
        backend.recover_ranks(extend_ranks)
    else:
        while "extend" not in signals:
            time.sleep(1)

        world = init_world_group(
            rank,
            num_processes,
            active_value=1,
            is_extension=True,
            max_world_size=num_processes,
        )
        world.process_group.join_group()

    # Ensure correct operation after extension
    tensor = paddle.to_tensor([rank + 1], dtype=paddle.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert tensor.item() == sum(range(1, num_processes + 1)), (
        f"Rank {rank} expected {sum(range(1, num_processes + 1))}, "
        f"get {tensor.item()}"
    )


def _deferred_recovery_worker(rank, num_processes, signals):
    """Worker for testing deferred rank recovery join."""
    if rank < num_processes:
        world = init_world_group(rank, num_processes, active_value=1)
        backend = world.process_group

        while backend.get_num_synced_ranks() < num_processes:
            time.sleep(0.1)
        if rank == broken_rank:
            return  # Simulate broken rank

        expected_without_broken = sum(range(0, num_processes)) - broken_rank
        tensor = paddle.to_tensor([rank], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == expected_without_broken

        time.sleep(5)
        signals["recover"] = 1
        while True:
            (peer_state,) = backend.get_peer_state([broken_rank])
            if peer_state:
                break

        # Healthy ranks keep making progress before the recovered rank joins.
        tensor = paddle.to_tensor([rank], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == expected_without_broken

        while "join_ready" not in signals:
            time.sleep(0.1)

        backend.recover_ranks([broken_rank])

        tensor = paddle.to_tensor([rank], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(0, num_processes)), (
            f"Rank {rank} expected {sum(range(0, num_processes))}, "
            f"get {tensor.item()}"
        )
    else:
        while "recover" not in signals:
            time.sleep(0.1)

        world = init_world_group(
            broken_rank,
            num_processes,
            device_id=broken_rank,
            active_value=1,
            is_extension=True,
        )
        backend = world.process_group

        # Deferred join starts in a local-only mode so collectives stay self-contained.
        tensor = paddle.to_tensor([broken_rank], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == broken_rank

        signals["join_ready"] = 1
        backend.join_group()

        tensor = paddle.to_tensor([broken_rank], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == sum(range(0, num_processes)), (
            f"Rank {rank} expected {sum(range(0, num_processes))}, "
            f"get {tensor.item()}"
        )


def _extension_world_plain_subgroup_worker(rank, num_processes, signals):
    """
    WORLD joins as an extension while its sub-group rendezvouses normally.

    Layout (num_processes=4): survivors=[0,1] hold WORLD, joiners=[2,3] restart
    and build sub-group [2,3]. Values are rank+1, so WORLD sums to 3 before the
    join and 10 after, and the sub-group sums to 7 throughout.
    """
    assert num_processes == 4, "this test assumes num_processes=4"

    survivor_world_size = num_processes // 2
    join_ranks = list(range(survivor_world_size, num_processes))

    survivor_sum = sum(range(1, survivor_world_size + 1))
    full_sum = sum(range(1, num_processes + 1))
    subgroup_sum = sum(r + 1 for r in join_ranks)

    if rank < survivor_world_size:
        world = init_world_group(
            rank, survivor_world_size, max_world_size=num_processes
        )
        backend = world.process_group

        tensor = paddle.to_tensor([rank + 1], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == survivor_sum, (
            f"Rank {rank} WORLD before join expected {survivor_sum}, got {tensor.item()}"
        )

        if rank == 0:
            signals["extend"] = 1
        while not all(backend.get_peer_state(join_ranks)):
            time.sleep(0.1)

        # Hold off recover_ranks until the joiners have finished a real
        # collective on their non-extension sub-group, so that sub-group cannot
        # be passing merely because WORLD had already been repaired.
        while "subgroup_ok" not in signals:
            time.sleep(0.1)

        backend.recover_ranks(join_ranks)
    else:
        while "extend" not in signals:
            time.sleep(0.1)

        world = init_world_group(
            rank,
            num_processes,
            active_value=1,
            is_extension=True,
            max_world_size=num_processes,
        )
        backend = world.process_group

        # An extension rank is local-only on WORLD until join_group().
        tensor = paddle.to_tensor([rank + 1], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == rank + 1, (
            f"Rank {rank} WORLD local-only expected {rank + 1}, got {tensor.item()}"
        )

        # The sub-group covers only the new ranks, so it is built without
        # is_extension and needs no join_group().
        sub_group = new_mooncake_group(join_ranks, gid=100)

        tensor = paddle.to_tensor([rank + 1], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=sub_group)
        assert tensor.item() == subgroup_sum, (
            f"Rank {rank} sub-group expected {subgroup_sum}, got {tensor.item()}"
        )

        sub_active = sub_group.process_group.get_active_ranks().tolist()
        assert sub_active == [1] * len(join_ranks), (
            f"Rank {rank} sub-group active ranks expected all active, got {sub_active}"
        )

        signals["subgroup_ok"] = 1
        backend.join_group()

        tensor = paddle.to_tensor([rank + 1], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor.item() == full_sum, (
            f"Rank {rank} WORLD after join expected {full_sum}, got {tensor.item()}"
        )

        # Repairing WORLD must not disturb the sub-group.
        tensor = paddle.to_tensor([rank + 1], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=sub_group)
        assert tensor.item() == subgroup_sum, (
            f"Rank {rank} sub-group after WORLD join expected {subgroup_sum}, "
            f"got {tensor.item()}"
        )
        return

    tensor = paddle.to_tensor([rank + 1], dtype=paddle.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert tensor.item() == full_sum, (
        f"Rank {rank} WORLD after join expected {full_sum}, got {tensor.item()}"
    )


class TestMooncakeBackend(unittest.TestCase):
    num_processes = 4

    @classmethod
    def setUpClass(cls) -> None:
        mp.set_start_method('spawn')

    def setUp(self) -> None:
        if device_count() < self.num_processes:
            self.skipTest(f"needs {self.num_processes} GPUs")

    def _spawn(self, worker, nprocs):
        mp_manager = mp.Manager()
        signals = mp_manager.dict()

        processes = []
        for rank in range(nprocs):
            p = mp.Process(
                target=worker,
                args=(rank, self.num_processes, signals)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(p.exitcode, 0)

    def test_elastic_extension(self):
        self._spawn(_elastic_worker, self.num_processes)

    def test_rank_recovery_deferred_join(self):
        self._spawn(_deferred_recovery_worker, self.num_processes + 1)

    def test_extension_world_with_plain_subgroup(self):
        self._spawn(_extension_world_plain_subgroup_worker, self.num_processes)


if __name__ == "__main__":
    unittest.main()
