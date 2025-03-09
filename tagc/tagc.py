import bisect
from   enum import IntEnum
import functools
import math
import numpy as np
from   typing import Callable, Optional

import torch
import torch.distributed as dist
from   torch.distributed.algorithms._comm_hooks import default_hooks

import api

# The block size of the data. We set the block size to 1024, which is the max number of thread
# in one GPU block. TODO does it depend on GPU?
BLOCK_SIZE = 1024


# TODO consider decreasing number of sparsification steps or getting rid of table by limiting
# number of bisection times
SPARSIFY_STEPS = [1e-20, 1e-15, 1e-10, 1e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5,
                  1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]


class IndexSize(IntEnum):
    ONE_BIT = 1
    FOUR_BITS = 4


def fp32_compress_hook(state: default_hooks.LowPrecisionState, grad: torch.Tensor, output: Optional[torch.Tensor] = None):
    fp32_hook = functools.partial(default_hooks._low_precision_hook, torch.float32)
    return fp32_hook(state, grad, output)


class HomomorphicCompressState(default_hooks.LowPrecisionState):

    __slots__ = [
        "num_processes",
        "process_index",
        "cuda_stream_nccl",
        "cuda_streams_sparse",
        "grad_num",
        "grad_remainders",
        "iter_num",
        "aligned_sparsified_grad_shards",
        "aligned_output",
        "sparse_readys",
        "indexes",
        "local_index_ready",
        "count_sketches",
        "count_mappings",
        "global_index_ready",
        "compress_ratio",
        "index_size",
        "sparsify_fraction",
    ]

    def __init__(
        self,
        process_group: int,
        num_processes: int,
        process_index: int,
        sparsify_fraction: float,
        index_size: IndexSize,
        compress_ratio: float,
    ):
        super().__init__(process_group)
        self.num_processes = num_processes
        self.process_index = process_index
        self.sparsify_fraction = sparsify_fraction
        self.index_size = index_size
        self.compress_ratio = compress_ratio

        self.cuda_stream_nccl = torch.cuda.Stream()
        self.cuda_streams_sparse = [torch.cuda.Stream() for _ in range(self.num_processes)]
        self.grad_num = 0
        self.grad_remainders = {}

        self.iter_num = 0

        self.aligned_sparsified_grad_shards = {}
        self.aligned_output = None
        self.sparse_readys = {}

        self.indexes = {}
        self.local_index_ready = {}
        self.count_sketches = {}
        self.count_mappings = {}

        self.global_index_ready = {}

    def next_grad(self):
        self.grad_num += 1

    def add_grad_remainder(self, grad: torch.Tensor):
        if self.grad_num in self.grad_remainders:
            grad += self.grad_remainders[self.grad_num]

    def replace_grad_remainder(self, grad_remainder: torch.Tensor):
        self.grad_remainders[self.grad_num] = grad_remainder

    def end_iteration(self):
        self.grad_num = 0
        self.iter_num += 1


def reduce_hook(state: default_hooks.DefaultState, grad: torch.Tensor, dst: int, skip_divide=False):
    r"""
    Implement the  FSDP communication hook for ``reduce`` algorithm and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
    # Average grad by pre-division factor. Together pre- and post-division factors
    # lead to an overall averaging by world_size, required for consistency with PyTorch DDP.
    # This is a two-step process to avoid potential underflow and overflow.
    if state.gradient_predivide_factor > 1 and not skip_divide:
        grad.div_(state.gradient_predivide_factor)
    dist.reduce(grad, dst, group=state.process_group)
    # Average grad by post-division factor.
    if state.gradient_postdivide_factor > 1 and not skip_divide:
        grad.div_(state.gradient_postdivide_factor)


def allreduce_hook(state: default_hooks.DefaultState, grad: torch.Tensor, skip_divide=False):
    r"""
    Implement the  FSDP communication hook for ``all_reduce`` algorithm and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
    # Average grad by pre-division factor. Together pre- and post-division factors
    # lead to an overall averaging by world_size, required for consistency with PyTorch DDP.
    # This is a two-step process to avoid potential underflow and overflow.
    if state.gradient_predivide_factor > 1 and not skip_divide:
        grad.div_(state.gradient_predivide_factor)
    dist.all_reduce(grad, group=state.process_group)
    # Average grad by post-division factor.
    if state.gradient_postdivide_factor > 1 and not skip_divide:
        grad.div_(state.gradient_postdivide_factor)


def sparsify(state: HomomorphicCompressState, grad: torch.Tensor, shard: int):
    state.add_grad_remainder(grad)

    def fraction_from_step(step: float):
        result = torch.where(torch.abs(grad) < step, 0., grad)
        result_fraction = 1 - result.count_nonzero().item() / result.numel()
        return result_fraction

    sparsify_step_idx = bisect.bisect_left(SPARSIFY_STEPS, state.sparsify_fraction, key=fraction_from_step)
    if sparsify_step_idx == 0:
        result = grad
    elif sparsify_step_idx < len(SPARSIFY_STEPS):
        result = torch.where(torch.abs(grad) < SPARSIFY_STEPS[sparsify_step_idx - 1], 0., grad)
    else:
        result = torch.where(torch.abs(grad) < SPARSIFY_STEPS[len(SPARSIFY_STEPS) - 1], 0., grad)
    state.replace_grad_remainder(grad - result)
    state.next_grad()
    return result


def align_sparsify_and_create_index(state: HomomorphicCompressState, shard : int,
                       full_grad: torch.Tensor, output: torch.Tensor):
    if shard in state.local_index_ready:
        return
    with torch.cuda.stream(state.cuda_streams_sparse[shard]):
        grad_shard = full_grad[shard * full_grad.numel() // state.num_processes:(shard + 1) * full_grad.numel() // state.num_processes]
        assert grad_shard.numel() == output.numel()

        aligned_grad_shard = grad_shard[:grad_shard.numel() // BLOCK_SIZE * BLOCK_SIZE]
        aligned_output = output[:grad_shard.numel() // BLOCK_SIZE * BLOCK_SIZE]

        aligned_sparsified_grad_shard = sparsify(state, aligned_grad_shard, shard)
        sparse_ready = torch.cuda.Event()
        sparse_ready.record()
        state.aligned_sparsified_grad_shards[shard] = aligned_sparsified_grad_shard
        state.aligned_output = aligned_output
        state.sparse_readys[shard] = sparse_ready

    with torch.cuda.stream(torch.cuda.default_stream()):
        state.sparse_readys[shard].wait()

    create_index(state, shard, aligned_sparsified_grad_shard, aligned_output)


def compress_aligned_sparsified_and_reduce_remainder(
        state: HomomorphicCompressState, shard : int,
        full_grad: torch.Tensor, output: torch.Tensor,
        aligned_sparsified_grad_shard: torch.Tensor, aligned_output: torch.Tensor):

    def pre_compute_next_shard():
        if shard < state.num_processes - 1:
            align_sparsify_and_create_index(state, shard + 1, full_grad, output)

    def pre_communicate_next_shard():
        if shard < state.num_processes - 1:
            all_reduce_index(state, shard + 1, full_grad, output)

    homomorphic_compress_shard_aligned_sparsified(state, shard, aligned_sparsified_grad_shard,
                                        aligned_output, pre_compute_next_shard,
                                        pre_communicate_next_shard)

    if full_grad.numel() % (state.num_processes * BLOCK_SIZE * BLOCK_SIZE) == 0:
        return

    with torch.cuda.stream(torch.cuda.default_stream()):
        grad_shard = full_grad[shard * full_grad.numel() // state.num_processes:(shard + 1) * full_grad.numel() // state.num_processes]
        grad_shard_remainder = grad_shard[grad_shard.numel() // BLOCK_SIZE * BLOCK_SIZE:]
        # Assuming slice provides view for actual tensor, not copying it
        # https://stackoverflow.com/questions/61964164/pytorch-tensor-slice-and-memory-usage
        output_remainder = output[output.numel() // BLOCK_SIZE * BLOCK_SIZE:]
        if shard == state.process_index:
            output_remainder.copy_(grad_shard_remainder)
        remainder_ready = torch.cuda.Event()
        remainder_ready.record()

    with torch.cuda.stream(state.cuda_stream_nccl):
        remainder_ready.wait()
        reduce_hook(state, output_remainder if shard == state.process_index else grad_shard_remainder, shard)


def homomorphic_compress_shard_not_aligned(state: HomomorphicCompressState, shard : int,
                                           full_grad: torch.Tensor, output: torch.Tensor):
    align_sparsify_and_create_index(state, shard, full_grad, output)
    compress_aligned_sparsified_and_reduce_remainder(
        state, shard, full_grad, output, state.aligned_sparsified_grad_shards[shard],
        state.aligned_output)


def create_index(state: HomomorphicCompressState, shard : int,
                 grad_shard: torch.Tensor, output: torch.Tensor):
    if shard in state.local_index_ready:
        return
    with torch.cuda.stream(torch.cuda.default_stream()):
        grid_size = (grad_shard.numel() + BLOCK_SIZE - 1)//BLOCK_SIZE
        if state.compress_ratio <= 1:
            state.local_index_ready[shard] = None
            return
        compressed_r = math.ceil(grid_size / state.compress_ratio)

        if state.index_size == 1:
            state.indexes[shard] = torch.zeros(grad_shard.numel()//8, dtype=torch.uint8, device=grad_shard.device)
        else:
            state.indexes[shard] = torch.zeros(grad_shard.numel()//2, dtype=torch.uint8, device=grad_shard.device)
        state.count_sketches[shard] = torch.zeros(compressed_r * BLOCK_SIZE, dtype=torch.float32, device=grad_shard.device)
        state.count_mappings[shard] = torch.zeros(compressed_r * BLOCK_SIZE, dtype=torch.uint8, device=grad_shard.device)

        if state.index_size == 1:
            api.torch_launch_create_index_1_bit(grad_shard, state.indexes[shard], grid_size, BLOCK_SIZE)
        else:
            api.torch_launch_create_index_4_bit(grad_shard, state.indexes[shard], grid_size, BLOCK_SIZE)
        state.local_index_ready[shard] = torch.cuda.Event()
        state.local_index_ready[shard].record()


def all_reduce_index(state: HomomorphicCompressState, shard : int,
                     grad_shard: torch.Tensor, output: torch.Tensor):
    if shard in state.global_index_ready:
        return
    if state.compress_ratio <= 1:
        if shard == state.process_index:
            output.copy_(grad_shard)
        reduce_hook(state, output if shard == state.process_index else grad_shard, shard)
        state.global_index_ready[shard] = None
        return

    with torch.cuda.stream(state.cuda_stream_nccl):
        state.local_index_ready[shard].wait()
        allreduce_hook(state, state.indexes[shard], skip_divide=True)
        state.global_index_ready[shard] = torch.cuda.Event()
        state.global_index_ready[shard].record()


def homomorphic_compress_shard_aligned_sparsified(state: HomomorphicCompressState, shard : int,
                                       grad_shard: torch.Tensor, output: torch.Tensor,
                                       network_time_callback: Callable[[],None],
                                       network_time_2_callback: Callable[[],None]):
    create_index(state, shard, grad_shard, output)

    all_reduce_index(state, shard, grad_shard, output)
    grid_size = (grad_shard.numel() + BLOCK_SIZE - 1)//BLOCK_SIZE
    compressed_r = math.ceil(grid_size / state.compress_ratio)

    network_time_callback()

    with torch.cuda.stream(torch.cuda.default_stream()):
        state.global_index_ready[shard].wait()
        # sets 1 to lowest bits of gradient -- this is used by compress
        if state.index_size == 1:
            api.torch_launch_read_index_1_bit(grad_shard, state.indexes[shard], grid_size, BLOCK_SIZE)
        else:
            api.torch_launch_read_index_4_bit(grad_shard, state.indexes[shard], grid_size, BLOCK_SIZE)

        api.torch_launch_compress_float_32(grad_shard, state.count_sketches[shard],
                                           state.count_mappings[shard], compressed_r, grid_size, BLOCK_SIZE)
        local_count_sketch_ready = torch.cuda.Event()
        local_count_sketch_ready.record()

    # TODO add saving diff in gradient and accumulating it
    with torch.cuda.stream(state.cuda_stream_nccl):
        local_count_sketch_ready.wait()
        reduce_hook(state, state.count_sketches[shard], shard)
        global_count_sketch_ready = torch.cuda.Event()
        global_count_sketch_ready.record()

    if shard != state.process_index:
        return

    network_time_2_callback()

#    FIXME TODO what is better, copy or zero? What is the difference in error/loss?
#    output.copy_(grad_shard)
    with torch.cuda.stream(torch.cuda.default_stream()):
        output.zero_()
        if state.index_size == 1:
            api.torch_launch_read_index_1_bit(output, state.indexes[shard], grid_size, BLOCK_SIZE)
        else:
            api.torch_launch_read_index_4_bit(output, state.indexes[shard], grid_size, BLOCK_SIZE)

        flag = torch.ones(1, dtype=torch.int32, device=grad_shard.device)
        flag_zero = torch.zeros(1, dtype=torch.int32, device=grad_shard.device)
        global_count_sketch_ready.wait()

        while flag[0] != flag_zero[0]:
            flag[0] = flag_zero[0]
            # As output is size of shard and count_sketch is created for shard, they should match in sizes
            api.torch_launch_decompress_float_32(output, state.count_sketches[shard], state.count_mappings[shard], compressed_r, grid_size, BLOCK_SIZE, flag)

        api.torch_launch_estimate_float_32(output, state.count_sketches[shard], compressed_r, grid_size, BLOCK_SIZE)


class TAGCState(HomomorphicCompressState):
    __slots__ = [
        "is_transformer_hook",
    ]

    def __init__(
        self,
        process_group: int,
        num_processes: int,
        process_index: int,
        sparsify_fraction: float,
        index_size: IndexSize,
        compress_ratio: float,
        is_transformer_hook: Callable[[int], bool],
    ):
        super().__init__(process_group, num_processes, process_index,
                         sparsify_fraction, index_size, compress_ratio)
        self.is_transformer_hook = is_transformer_hook


def transformer_compress_hook(state: TAGCState, grad: torch.Tensor, output: Optional[torch.Tensor] = None):
    if output is None:
        return default_hooks.allreduce_hook(state, grad)

    if not state.is_transformer_hook(grad.numel()):
        return homomorphic_compress_hook(state, grad, output)
    fp32_compress_hook(state, grad, output)


def homomorphic_compress_hook(state: HomomorphicCompressState, grad: torch.Tensor, output: Optional[torch.Tensor] = None):
    if output is None:
        return default_hooks.allreduce_hook(state, grad)

    # For easy handling in CUDA of non-adjusted grads
    # padding
    assert len(grad.shape) == 1 # gradient should be flattened

    comm_hook_gradient_ready = torch.cuda.Event()
    comm_hook_gradient_ready.record()

    state.local_index_ready = {}
    state.global_index_ready = {}

    for i in range(state.num_processes):
        with torch.cuda.stream(state.cuda_streams_sparse[i]):
            comm_hook_gradient_ready.wait()

    for i in range(state.num_processes):
        homomorphic_compress_shard_not_aligned(state, i, grad, output)
