# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.cuda
from torch.autograd import Function

from numba import cuda, jit, prange


# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: 序列的长度（假设两个输入的大小相同）
    :param n_passes: 2 * seq_len - 1（反对角线的数量）
    """
    # 每个块处理一对示例
    b = cuda.blockIdx.x
    # 我们的线程数与seq_len相同，因为我们所需的最多线程数
    # 等于最大反对角线上的元素数量
    tid = cuda.threadIdx.x

    inv_gamma = 1.0 / gamma

    # 遍历每个反对角线。只处理落在当前反对角线上的线程
    for p in range(n_passes):

        # 索引实际上是'p - tid'，但需要强制使其在边界内
        J = max(0, min(p - tid, max_j - 1))

        # 为了简化，我们定义从1开始的i, j（相对于I, J的偏移）
        i = tid + 1
        j = J + 1

        # 仅当元素[i, j]在当前反对角线上且在边界内时才计算
        if tid + J == p and (tid < max_i and J < max_j):
            # 如果在带宽外则不计算
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # 等待此块中的其他线程
        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # 索引逻辑与上述相同，但是反对角线需要
    # 反向进行

    for p in range(n_passes):
        # 反转顺序使循环向后进行
        rev_p = n_passes - p - 1

        # 将tid转换为I, J，然后是i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = tid + 1
        j = J + 1

        # 仅当元素[i, j]在当前反对角线上且在边界内时才计算
        if tid + J == rev_p and (tid < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # 如果在带宽外则不计算
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # 等待此块中的其他线程
        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA实现灵感来源于https://ieeexplore.ieee.org/document/8400444中提出的对角线方法:
    "在时间序列数据中开发模式发现方法及其GPU加速"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # 准备输出数组
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # 运行CUDA内核。
        # 设置CUDA的网格大小等于批处理大小（每个CUDA块处理一对样本）
        # 设置CUDA块大小等于较长序列的长度（等于最大对角线的大小）
        compute_softdtw_cuda[B, threads_per_block](
            cuda.as_cuda_array(D.detach()), gamma.item(), bandwidth.item(), N, M, n_passes, cuda.as_cuda_array(R)
        )
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1 : N + 1, 1 : M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](
            cuda.as_cuda_array(D_),
            cuda.as_cuda_array(R),
            1.0 / gamma.item(),
            bandwidth.item(),
            N,
            M,
            n_passes,
            cuda.as_cuda_array(E),
        )
        E = E[:, 1 : N + 1, 1 : M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# 以下是基于https://github.com/Sleepwalking/pytorch-softdtw的CPU实现
# 版权归Kanru Hua所有。
# 我添加了对批处理和剪枝的支持。
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1 : N + 1, 1 : M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1 : N + 1, 1 : M + 1]


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    基于https://github.com/Sleepwalking/pytorch-softdtw的CPU实现
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    可选择支持CUDA的软DTW实现
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        使用提供的参数初始化新实例
        :param use_cuda: 指示是否应使用CUDA实现的标志
        :param gamma: sDTW的gamma参数
        :param normalize: 指示是否执行归一化的标志
                          (如在https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790中讨论的)
        :param bandwidth: 用于剪枝的Sakoe-Chiba带宽。传递'None'将禁用剪枝。
        :param dist_func: 可选的逐点距离函数。如果为'None'，则将使用默认的欧几里得距离函数。
        """
        super().__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        检查输入并选择要使用的适当实现。
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
            print(
                "SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)"
            )
            use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        计算x和y中每个元素在每个时间步上的欧几里得距离
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        计算X和Y之间的软DTW值
        :param X: 一批示例, batch_size x seq_len x dims
        :param Y: 另一批示例, batch_size x seq_len x dims
        :return: 计算结果
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)


# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    通过sdtw运行a和b，并计时前向和反向传递。
    假设a需要梯度。
    :return: 计时, 前向结果, 反向结果
    """

    from timeit import default_timer as timer

    # 前向传递
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # 反向传递
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # 总时间
    t += end - start

    return t, forward, grads


# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print(
        "为batch_size={}, seq_len_a={}, seq_len_b={}, dims={}分析forward() + backward()时间...".format(
            batch_size, seq_len_a, seq_len_b, dims
        )
    )

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # 验证结果
        assert torch.allclose(forward_cpu, forward_gpu.cpu())
        assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if (
            i > 0
        ):  # 忽略第一次运行，以防这是冷启动（因为脚本冷启动时计时不准）
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    # 平均并记录
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    print("  加速比: ", avg_cpu / avg_gpu)
    print()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    torch.manual_seed(1234)

    profile(128, 17, 15, 2, tol_backward=1e-6)
    profile(512, 64, 64, 2, tol_backward=1e-4)
    profile(512, 256, 256, 2, tol_backward=1e-3)
