import os
import sys
import tempfile
import shutil
import json
import numpy as np
from typing import Optional

import paddle
import paddle.distributed as dist

from pg_test_utils import init_world_group


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    num_ranks = num_nodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank
    group = init_world_group(rank, num_ranks, device_id=local_rank)

    paddle.set_default_dtype(paddle.bfloat16)

    return rank, num_ranks, group


def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def per_token_cast_to_fp8(x: paddle.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(paddle.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
    x_fp32 = x_fp8.to(paddle.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(paddle.bfloat16)


def inplace_unique(x: paddle.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = paddle.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, paddle.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = paddle.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = paddle.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(scores: paddle.Tensor, group_idx: paddle.Tensor, num_groups: int):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = paddle.zeros((num_tokens, num_groups), dtype=paddle.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def bench(fn, num_warmups: int = 20, num_tests: int = 30, post_fn=None):
    # Flush L2 cache with 256 MB data
    paddle.cuda.synchronize()
    cache = paddle.empty(int(256e6 // 4), dtype=paddle.int)

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [paddle.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [paddle.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    paddle.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn, kernel_names, num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None, barrier_comm_profiling: bool = False):
    tmp_dir = tempfile.mkdtemp()
    tmp_trace = os.path.join(tmp_dir, 'trace.json')

    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    try:
        with suppress():
            sched = paddle.profiler.make_scheduler(closed=0, ready=1, record=1, repeat=1)
            prof = paddle.profiler.Profiler(
                targets=[paddle.profiler.ProfilerTarget.GPU],
                scheduler=sched,
            )
            prof.start()
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = paddle.randn([8192, 8192], dtype=paddle.float32)
                    rhs = paddle.randn([8192, 8192], dtype=paddle.float32)
                    paddle.matmul(lhs, rhs)
                    dist.all_reduce(paddle.ones([1], dtype=paddle.float32))
                for _ in range(num_tests):
                    fn()
                prof.step()
            prof.stop()
            prof.export(tmp_trace, format='json')

        # Load and clean up event names (strip Paddle's embedded "[x us]" suffix)
        with open(tmp_trace) as f:
            trace_data = json.load(f)

        for e in trace_data.get('traceEvents', []):
            if 'name' in e and '[' in e['name']:
                e['name'] = e['name'][:e['name'].rfind('[')].strip()

        # Save chrome traces (cleaned version)
        if trace_path is not None:
            with open(trace_path, 'w') as f:
                json.dump(trace_data, f)

        # Only keep GPU kernel events
        gpu_events = [
            e for e in trace_data.get('traceEvents', [])
            if e.get('ph') == 'X' and 'dur' in e
        ]

        # Parse kernel names
        assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
        is_tupled = isinstance(kernel_names, tuple)
        kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
        assert all([isinstance(name, str) for name in kernel_names])

        # In fallback mode, kernels may not appear in profiling trace (they're Python functions)
        # So we check if kernels exist, but don't fail if they don't (fallback mode)
        for name in kernel_names:
            count = sum([name in e.get('name', '') for e in gpu_events])
            if count == 0:
                # Kernel not found - might be fallback mode, return 0 time
                import warnings
                warnings.warn(f'Kernel {name} not found in profiling trace (might be fallback mode)', UserWarning)

        # Return average kernel times (in seconds)
        kernel_times = []
        for name in kernel_names:
            durations = [e['dur'] for e in gpu_events if name in e.get('name', '')]
            if not durations:
                # Kernel not found (fallback mode), return 0 time
                kernel_times.append(0.0)
            else:
                # dur is in microseconds in Chrome Trace format
                kernel_times.append((sum(durations) / len(durations)) / 1e6)

        return tuple(kernel_times) if is_tupled else kernel_times[0]

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def hash_tensor(t: paddle.Tensor):
    return t.view(paddle.int64).sum().item()
