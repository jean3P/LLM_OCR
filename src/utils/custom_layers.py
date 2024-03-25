import copy
import functools
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.activations import ACT2FN
from typing import Dict, Any
from hqq.core.quantize import HQQLinear, Quantizer

import torch
from torch import nn
from torch.nn import functional as F

from .packing import pack_4bit_u8_common, pack_2bit_u8_common, unpack_4bit_u8_common, unpack_2bit_u8_common

import triton
import triton.language as tl
import torch
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': N,
                       'BLOCK_SIZE_K': K, 'GROUP_SIZE_M': 1},
                      num_stages=S, num_warps=W) for N, K, S, W in
        [
            #             (32, 16, 1, 2),
            (32, 32, 4, 4),
            #             (32, 32, 5, 2),
            #             (32, 32, 5, 8),
            #             (32, 128, 2, 4),
            #             (64, 32, 2, 4),
            #             (64, 32, 3, 4),
            #             (64, 32, 4, 4),
            #             (64, 32, 4, 8),
            #             (64, 32, 5, 2),
            #             (64, 32, 5, 8),
            #             (64, 64, 3, 8),
            #             (128, 32, 2, 8),
            #             (128, 32, 3, 4),
            #             (128, 32, 3, 8),
            #             (128, 32, 4, 4),
            #             (128, 32, 4, 8),
            #             (256, 32, 3, 8),
            #             (256, 32, 4, 4),
            #             (256, 64, 3, 8),
        ]

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul4_kernel_transpose(
        a_ptr, b_ptr, c_ptr,
        scales_ptr, zeros_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_scales_g, stride_scales_n,
        stride_zeros_g, stride_zeros_n,
        groupsize, NO_GROUPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (N//2, K) int32
    C is of shape (M, N) float16
    scales is of shape (G, K) float16
    zeros is of shape (G, K) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is N // groupsize.
    Set NO_GROUPS to groupsize == N, in which case G = 1 and the kernel is more efficient.

    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group  #
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the N axis 2 times
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + (offs_bn[None, :] // 2) * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

    G = N // groupsize
    scales_ptrs = scales_ptr + (offs_bn[None, :] % G) * stride_scales_g  # (1, BLOCK_SIZE_N)
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] % G) * stride_zeros_g  # (1, BLOCK_SIZE_N)

    # shifter is used to extract the 4 bits of each element in the 8-bit word from B
    shifter = ((offs_bn + 1) % 2) * 4

    # If G == 1, scales and zeros are the same for all N, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32

    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        if not NO_GROUPS:
            offs_k_scale = BLOCK_SIZE_K * k + offs_k
            ptr = scales_ptrs + offs_k_scale[:, None] * stride_scales_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            scales = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            ptr = zeros_ptrs + offs_k_scale[:, None] * stride_zeros_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        # Now we need to unpack b (which is 4-bit values) into 8-bit values
        b = (b >> shifter[None, :]) & 0xF  # Extract the 4-bit values
        b = b.to(tl.float16)
        b = (b - zeros) * scales  # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul4_transpose(groupsize: int, a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor,
                             zeros: torch.FloatTensor, bias: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (N//2, K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize

    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """
    assert a.shape[-1] == (qweight.shape[1])
    assert a.is_contiguous(), "A must be contiguous"
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]

    # Flatten a into (-1, K)
    x = a.view(-1, a.shape[-1])

    M, K = x.shape
    N = qweight.shape[0] * 2
    # This is based on the possible BLOCK_SIZE_Ks
    #     assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
    # This is based on the possible BLOCK_SIZE_Ns
    #     assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0 and N % 128 == 0 and N % 256 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
    # This is based on the possible BLOCK_SIZE_Ks
    #     assert groupsize % 32 == 0 and groupsize % 64 == 0 and groupsize % 128 == 0, "groupsize must be a multiple of 32, 64, and 128"

    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul4_kernel_transpose[grid](
        x, qweight, c,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        groupsize, groupsize == N,
    )

    # Reshape c
    c = c.view(a.shape[:-1] + (N,))  # (..., N)

    # Add bias
    if bias is not None:
        c = c + bias

    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': N,
                       'BLOCK_SIZE_K': K, 'GROUP_SIZE_M': 1},
                      num_stages=S, num_warps=W) for N, K, S, W in
        [
            #             (32, 16, 1, 2),
            (32, 32, 4, 4),  # best
            #             (32, 32, 5, 2),
            #             (32, 32, 5, 8),
            #             (32, 128, 2, 4),
            #             (64, 32, 2, 4),
            #             (64, 32, 3, 4),
            #             (64, 32, 4, 4),
            #             (64, 32, 4, 8),
            #             (64, 32, 5, 2),
            #             (64, 32, 5, 8),
            #             (64, 64, 3, 8),
            #             (128, 32, 2, 8),
            #             (128, 32, 3, 4),
            #             (128, 32, 3, 8),
            #             (128, 32, 4, 4),
            #             (128, 32, 4, 8),
            #             (256, 32, 3, 8),
            #             (256, 32, 4, 4),
            #             (256, 64, 3, 8),
        ]

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul2_kernel_transpose(
        a_ptr, b_ptr, c_ptr,
        scales_ptr, zeros_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_scales_g, stride_scales_n,
        stride_zeros_g, stride_zeros_n,
        groupsize, NO_GROUPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (N // 4, K) int8
    C is of shape (M, N) float16
    scales is of shape (G, K) float16
    zeros is of shape (G, K) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is N // groupsize.
    Set NO_GROUPS to groupsize == N, in which case G = 1 and the kernel is more efficient.

    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group  #
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the N axis 4 times
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + (offs_bn[None, :] // 4) * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

    G = N // groupsize
    scales_ptrs = scales_ptr + (offs_bn[None, :] % G) * stride_scales_g  # (1, BLOCK_SIZE_N)
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] % G) * stride_zeros_g  # (1, BLOCK_SIZE_N)

    # shifter is used to extract the 2 bits of each element in the 8-bit word from B
    shifter = (3 - (offs_bn % 4)) * 2

    # If G == 1, scales and zeros are the same for all N, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,)

    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        if not NO_GROUPS:
            offs_k_scale = BLOCK_SIZE_K * k + offs_k
            ptr = scales_ptrs + offs_k_scale[:, None] * stride_scales_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            scales = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            ptr = zeros_ptrs + offs_k_scale[:, None] * stride_zeros_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        # Now we need to unpack b (which is 4-bit values) into 8-bit values
        b = (b >> shifter[None, :]) & 0b11  # Extract the 2-bit values
        b = b.to(tl.float16)
        b = (b - zeros) * scales  # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul2_transpose(groupsize: int, a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor,
                             zeros: torch.FloatTensor, bias: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (N // 4, K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize

    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """

    assert a.shape[-1] == (qweight.shape[1])
    assert a.is_contiguous(), "A must be contiguous"
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]

    # Flatten a into (-1, K)
    x = a.view(-1, a.shape[-1])

    M, K = x.shape
    N = qweight.shape[0] * 4
    # This is based on the possible BLOCK_SIZE_Ks
    #     assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
    # This is based on the possible BLOCK_SIZE_Ns
    #     assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0 and N % 128 == 0 and N % 256 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
    # This is based on the possible BLOCK_SIZE_Ks
    #     assert groupsize % 32 == 0 and groupsize % 64 == 0 and groupsize % 128 == 0, "groupsize must be a multiple of 32, 64, and 128"

    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul2_kernel_transpose[grid](
        x, qweight, c,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        groupsize, groupsize == N,
    )

    # Reshape c
    c = c.view(a.shape[:-1] + (N,))  # (..., N)

    # Add bias
    if bias is not None:
        c = c + bias

    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': N,
                       'BLOCK_SIZE_K': K, 'GROUP_SIZE_M': 1},
                      num_stages=S, num_warps=W) for N, K, S, W in
        [
            #             (32, 16, 1, 2),
            #             (32, 32, 4, 4),
            #             (32, 32, 5, 2),
            (32, 32, 5, 8),  # best
            #             (32, 128, 2, 4),
            #             (64, 32, 2, 4),
            #             (64, 32, 3, 4),
            #             (64, 32, 4, 4),
            #             (64, 32, 4, 8),
            #             (64, 32, 5, 2),
            #             (64, 32, 5, 8),
            #             (64, 64, 3, 8),
            #             (128, 32, 2, 8),
            #             (128, 32, 3, 4),
            #             (128, 32, 3, 8),
            #             (128, 32, 4, 4),
            #             (128, 32, 4, 8),
            #             (256, 32, 3, 8),
            #             (256, 32, 4, 4),
            #             (256, 64, 3, 8),
        ]

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul3_kernel_transpose(
        a_ptr, b_ptr, c_ptr,
        scales_ptr, zeros_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_scales_g, stride_scales_n,
        stride_zeros_g, stride_zeros_n,
        groupsize, NO_GROUPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (ceil(N / 10), K) int32
    C is of shape (M, N) float16
    scales is of shape (G, K) float16
    zeros is of shape (G, K) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is N // groupsize.
    Set NO_GROUPS to groupsize == N, in which case G = 1 and the kernel is more efficient.

    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group  #
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)

    # b_ptrs is set up such that it repeats elements along the N axis 10 times
    b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + (offs_bn[None, :] // 10) * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

    G = N // groupsize
    scales_ptrs = scales_ptr + (offs_bn[None, :] % G) * stride_scales_g  # (1, BLOCK_SIZE_N)
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] % G) * stride_zeros_g  # (1, BLOCK_SIZE_N)

    # shifter is used to extract the 3 bits of each element in the 32-bit word from B
    shifter = (9 - (offs_bn % 10)) * 3

    # If G == 1, scales and zeros are the same for all N, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,)

    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        if not NO_GROUPS:
            offs_k_scale = BLOCK_SIZE_K * k + offs_k
            ptr = scales_ptrs + offs_k_scale[:, None] * stride_scales_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            scales = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            ptr = zeros_ptrs + offs_k_scale[:, None] * stride_zeros_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        # Now we need to unpack b (which is 3-bit values into 32-bit values)
        b = (b >> shifter[None, :]) & 0b111  # Extract the 3-bit values
        b = b.to(tl.float16)
        b = (b - zeros) * scales  # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul3_transpose(groupsize: int, a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor,
                             zeros: torch.FloatTensor, N: int,
                             bias: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (ceil(N / 10), K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize

    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """

    assert a.shape[-1] == (qweight.shape[1])
    assert a.is_contiguous(), "A must be contiguous"
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]

    # Flatten a into (-1, K)
    x = a.view(-1, a.shape[-1])

    M, K = x.shape
    assert 0 <= (qweight.shape[0] * 10 - N) < 10

    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul3_kernel_transpose[grid](
        x, qweight, c,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        groupsize, groupsize == N,
    )

    # Reshape c
    c = c.view(a.shape[:-1] + (N,))  # (..., N)

    # Add bias
    if bias is not None:
        c = c + bias

    return c


class HQQLinearTritonSavable(HQQLinear):
    def __init__(self, layer, quant_config, meta=None, **kwargs):
        """
        Example how to get meta:
        >>>> meta1 = HQQLinearSavable.get_hqq_meta((hidden_dim, ffn_dim), quant_config)
        >>>> meta2 = HQQLinearSavable.get_hqq_meta((ffn_dim, hidden_dim), quant_config)
        """

        assert quant_config['weight_quant_params']['nbits'] in [2, 3, 4]

        super().__init__(layer, quant_config, **kwargs)

        if not hasattr(self, 'meta'):
            assert meta is not None
            self.meta = copy.deepcopy(meta)

        self._register_state_dict_hook(self._add_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_from_state_dict_hook)

    def quantize(self, *args, **kwargs):
        super().quantize(*args, **kwargs)

        # repacking
        self.repack()

    def repack(self):
        if self.W_q.shape != self.meta['shape']:
            W_q = Quantizer.unpack[self.meta['packing']](self.W_q)
            sh = self.meta['shape']
            W_q = W_q.reshape((-1,) + sh[1:])
            W_q = W_q[:sh[0], ...]
            self.W_q = Quantizer.pack[self.meta['packing']](W_q)

    def forward(self, x):
        return self.forward_triton(x)

    def set_backend(self, backend):
        pass

    @torch.inference_mode()
    def forward_triton(self, x):
        assert self.ready, "model was not quantized"
        assert self.meta['axis'] == 0

        W_q, meta = self.W_q, self.meta

        del_keys = []
        if 'quant_scale' in meta and meta['quant_scale']:
            meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']);
            del_keys.append('scale')
        if 'quant_zero' in meta and meta['quant_zero']:
            meta['zero'] = Quantizer.dequantize(meta['zero_q'], meta['meta_zero']);
            del_keys.append('zero')

        K = meta['shape'][1]
        N = meta['shape'][0]

        if self.meta['nbits'] == 4:
            fn = triton_matmul4_transpose
        elif self.meta['nbits'] == 3:
            fn = functools.partial(triton_matmul3_transpose, N=N)
        elif self.meta['nbits'] == 2:
            fn = triton_matmul2_transpose
        else:
            raise RuntimeError(f"nbits == {self.meta['nbits']} isn't yet supported")

        output = fn(
            meta['group_size'], x,
            W_q.view(-1, K),
            meta['scale'].view(-1, K),
            meta['zero'].view(-1, K),
            bias=self.bias if hasattr(self, 'bias') else None,
        )

        # Cleanup
        for key in del_keys:
            del meta[key]

        return output

    # to support .forward_pytorch(...) - backward compatibility
    @torch.inference_mode()
    def dequantize(self):
        assert self.ready, "model was not quantized"
        W_q, meta = self.W_q, self.meta
        del_keys = []
        if (meta['quant_scale']):
            meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']);
            del_keys.append('scale')
        if (meta['quant_zero']):
            meta['zero'] = Quantizer.dequantize(meta['zero_q'], meta['meta_zero']);
            del_keys.append('zero')

        W_q_p = Quantizer.unpack[meta['packing']](W_q).half()
        W_q_p = W_q_p[:meta['shape'][0], ...]
        W_q_p = W_q_p.reshape((meta['group_size'], -1))

        if ((meta['group_size'] is not None) and (meta['nbits'] == 3)):
            W_q_p = W_q_p[:meta['group_size']] if (meta['axis'] == 0) else W_q_p[:, :meta['group_size']]
        W_est = ((W_q_p - meta['zero']) * meta['scale']).reshape(meta['shape'])

        # Cleanup
        del W_q_p
        for key in del_keys: del meta[key]
        return W_est

    @classmethod
    def get_hqq_meta(cls, linear_shape, quant_config):
        layer = HQQLinear(nn.Linear(*linear_shape, bias=False), quant_config)
        meta = layer.meta

        def _remove_tensors_recursive(d):
            keys = list(d.keys())

            for k in keys:
                if isinstance(d[k], torch.Tensor):
                    del d[k]
                elif isinstance(d[k], dict):
                    _remove_tensors_recursive(d[k])

        _remove_tensors_recursive(meta)

        return meta

    @staticmethod
    def _add_to_state_dict_hook(self, state_dict, prefix, local_metadata):
        tensor_paths = self._get_tensor_paths(self.meta)
        assert set(tensor_paths).issubset(
            {'scale_q', 'meta_scale.scale', 'meta_scale.zero', 'zero_q', 'meta_zero.scale', 'meta_zero.zero',
             'scale', 'zero'}
        )

        def _add(name, value):
            state_dict[prefix + name] = value

        _add('W_q', self.W_q)

        if self.bias is not None:
            _add('bias', self.bias)

        if 'meta_scale' in self.meta:
            _add('meta.scale_q', self.meta['scale_q'])
            _add('meta.meta_scale.scale', self.meta['meta_scale']['scale'])
            _add('meta.meta_scale.zero', self.meta['meta_scale']['zero'])
        else:
            _add('meta.scale', self.meta['scale'])

        if 'meta_zero' in self.meta:
            _add('meta.zero_q', self.meta['zero_q'])
            _add('meta.meta_zero.scale', self.meta['meta_zero']['scale'])
            _add('meta.meta_zero.zero', self.meta['meta_zero']['zero'])
        else:
            _add('meta.zero', self.meta['zero'])

        return state_dict

    def _load_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                   error_msgs):
        tensor_paths = [k[len(prefix + 'meta.'):] for k in state_dict.keys() if k.startswith(prefix + 'meta.')]
        assert set(tensor_paths).issubset(
            {'scale_q', 'meta_scale.scale', 'meta_scale.zero', 'zero_q', 'meta_zero.scale', 'meta_zero.zero',
             'scale', 'zero'}
        )

        def _del(name):
            del state_dict[prefix + name]

        def _set(name):
            setattr(self, name, state_dict[prefix + name])
            _del(name)

        def _get(name):
            v = state_dict[prefix + name]
            _del(name)
            return v

        _set('W_q')
        if 'bias' in state_dict:
            _set('bias')
        else:
            self.bias = None

        if not hasattr(self, 'meta'):
            self.meta = {}

        if (prefix + 'meta.meta_scale.scale') in state_dict:
            self.meta['scale_q'] = _get('meta.scale_q')
            self.meta['quant_scale'] = True
            if not 'meta_scale' in self.meta:
                self.meta['meta_scale'] = {}
            self.meta['meta_scale'] |= {
                'scale': _get('meta.meta_scale.scale'),
                'zero': _get('meta.meta_scale.zero')
            }
        else:
            self.meta['scale'] = _get('meta.scale')
        if (prefix + 'meta.meta_zero.scale') in state_dict:
            self.meta['zero_q'] = _get('meta.zero_q')
            self.meta['quant_zero'] = True
            if not 'meta_zero' in self.meta:
                self.meta['meta_zero'] = {}
            self.meta['meta_zero'] |= {
                'scale': _get('meta.meta_zero.scale'),
                'zero': _get('meta.meta_zero.zero')
            }
        else:
            self.meta['zero'] = _get('meta.zero')
        self.ready = True

        # self.cuda()
        # self.in_gpu = self.W_q.device.type == 'cuda'
        # assert self.in_gpu

        self.repack()

    @classmethod
    def _get_tensor_paths(cls, state: Dict[str, Any], prefix=''):
        paths = []

        for k, v in state.items():
            if isinstance(v, dict):
                paths += cls._get_tensor_paths(v, prefix=k + '.')
            elif isinstance(v, torch.Tensor):
                paths.append(prefix + k)

        return paths

    def state_dict(self, *args, **kwargs):
        return nn.Module.state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        nn.Module.load_state_dict(self, *args, **kwargs)


class MixtralBLockSparseTop2MLP_HQQ(nn.Module):
    def __init__(self, config: MixtralConfig, quant_config: Dict[str, Any], meta1, meta2):
        super().__init__()

        self.w1 = HQQLinearTritonSavable(None, quant_config, meta1)
        self.w2 = HQQLinearTritonSavable(None, quant_config, meta2)
        self.w3 = HQQLinearTritonSavable(None, quant_config, meta1)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class SparseMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = gate
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        active_experts = selected_experts.flatten().unique().tolist()

        # Loop over all available experts in the model and perform the computation on each expert
        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
                *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            idx, top_x = torch.where(expert_mask[expert_idx])
            assert top_x.shape[0] > 0

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits