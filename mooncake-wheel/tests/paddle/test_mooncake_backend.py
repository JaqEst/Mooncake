import os
import time
import unittest
import multiprocessing as mp

import paddle
paddle.enable_compat(scope={"mooncake"})
import mooncake.distributed as dist
from mooncake import pg


def worker(rank, world_size, results, collective):
    paddle.cuda.set_device(f'cuda:{rank}')

    dist.init_process_group(
        backend="mooncake",
        rank=rank,
        world_size=world_size,
        pg_options=pg.MooncakeBackendOptions(paddle.zeros((world_size,), dtype=paddle.int32)),
    )

    if collective == "all_reduce_sum":
        tensor = paddle.tensor([rank + 1], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        results[rank] = tensor.item()

    elif collective == "all_reduce_product":
        tensor = paddle.tensor([2], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT)
        results[rank] = tensor.item()

    elif collective == "all_reduce_min":
        tensor = paddle.tensor([rank + 10], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        results[rank] = tensor.item()

    elif collective == "all_reduce_max":
        tensor = paddle.tensor([rank + 10], dtype=paddle.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        results[rank] = tensor.item()

    elif collective == "all_gather":
        tensor = paddle.tensor([rank])
        gathered = [paddle.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        results[rank] = [t.item() for t in gathered]

    elif collective == "barrier":
        dist.barrier()
        results[rank] = "ok"

    elif collective == "gather":
        tensor = paddle.tensor([rank], dtype=paddle.int32)
        if rank == 0:
            gather_list = [paddle.zeros_like(tensor) for _ in range(world_size)]
            dist.gather(tensor, gather_list, dst=0)
            results[rank] = [t.item() for t in gather_list]
        else:
            dist.gather(tensor, dst=0)
            results[rank] = None

    elif collective == "scatter":
        tensor = paddle.zeros(1, dtype=paddle.int32)
        if rank == 0:
            scatter_list = [paddle.tensor([i], dtype=paddle.int32) for i in range(world_size)]
            dist.scatter(tensor, scatter_list, src=0)
        else:
            dist.scatter(tensor, src=0)
        results[rank] = tensor.item()

    elif collective == "reduce":
        tensor = paddle.tensor([1], dtype=paddle.int32)
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            results[rank] = tensor.item()
        else:
            results[rank] = None

    else:
        raise ValueError(f"Unsupported collective: {collective}")

    while len(results) < world_size:
        time.sleep(1)

    dist.destroy_process_group()


class TestMooncakeBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method('spawn')

    def setUp(self):
        self.world_size = paddle.cuda.device_count()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

    def tearDown(self):
        pass

    def _spawn_and_check(self, collective, expected_fn):
        mp_manager = mp.Manager()
        results = mp_manager.dict()

        processes = []
        for rank in range(self.world_size):
            p = mp.Process(
                target=worker,
                args=(rank, self.world_size, results, collective)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        expected = expected_fn(self.world_size)
        for r in range(self.world_size):
            self.assertEqual(results[r], expected)

    def test_allreduce_sum(self):
        # Expected sum = 1 + 2 + 3 + 4 = 10
        self._spawn_and_check("all_reduce_sum", lambda size: sum(range(1, size + 1)))

    def test_allreduce_product(self):
        # Expected product(2, 2, 2, ……, 2) = 2 ^ N
        self._spawn_and_check("all_reduce_product", lambda size: 2 ** size)

    def test_allreduce_min(self):
        # Expected Min(10, 11, ……) = 10
        self._spawn_and_check("all_reduce_min", lambda size: 10)

    def test_allreduce_max(self):
        # Expected Max(10, 11, ..., 10+N-1) = 10 + N - 1
        self._spawn_and_check("all_reduce_max", lambda size: 10 + size - 1)

    def test_allgather(self):
        # Expected gather = [0, 1, 2, 3]
        self._spawn_and_check("all_gather", lambda size: list(range(size)))

    def test_barrier(self):
        self._spawn_and_check("barrier", lambda size: "ok")

    def test_gather(self):
        # Expected gather (Root) = [0, 1, 2, ..., size-1]
        # Expected gather (Others) = None
        mp_manager = mp.Manager()
        results = mp_manager.dict()

        processes = []
        for rank in range(self.world_size):
            p = mp.Process(
                target=worker,
                args=(rank, self.world_size, results, "gather")
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.assertEqual(results[0], list(range(self.world_size)))
        for r in range(1, self.world_size):
            self.assertIsNone(results[r])

    def test_scatter(self):
        # Expected scatter (Rank i) = i
        mp_manager = mp.Manager()
        results = mp_manager.dict()

        processes = []
        for rank in range(self.world_size):
            p = mp.Process(
                target=worker,
                args=(rank, self.world_size, results, "scatter")
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for r in range(self.world_size):
            self.assertEqual(results[r], r)

    def test_reduce(self):
        # Expected reduce (Root) = sum([1, 1, ..., 1]) = size
        # Expected reduce (Others) = None
        mp_manager = mp.Manager()
        results = mp_manager.dict()

        processes = []
        for rank in range(self.world_size):
            p = mp.Process(
                target=worker,
                args=(rank, self.world_size, results, "reduce")
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.assertEqual(results[0], self.world_size)
        for r in range(1, self.world_size):
            self.assertIsNone(results[r])


if __name__ == "__main__":
    unittest.main()
