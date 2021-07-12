import numba as nb
from numba import cuda
import numpy as np
import taichi as tc
import time
################################################################################
@nb.jit(nopython = True, nogil = True)
def matmul_CPU(A, B, C):
    #
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

################################################################################
@cuda.jit
def matmul_CUDA(A, B, C):
    #
    j = cuda.threadIdx.x
    i = cuda.blockIdx.x
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

################################################################################
@tc.kernel
def matmul_TaiChi(d: tc.i64):
    #
    for i, j in C_tc:
        tmp = 0.
        for k in range(d):
            tmp += A_tc[i, k] * B_tc[k, j]
        C_tc[i, j] = tmp

################################################################################
def main():
    #
    n = 4096
    d = 4096
    m = 1024
    A = np.random.rand(n, d)
    B = np.random.rand(d, m)
    #
    C = np.zeros((n, m))
    start_time = time.perf_counter()
    C = np.dot(A, B)
    end_time = time.perf_counter()
    print('Total Time with Numpy dot: %fs.' % (end_time - start_time))
    print(C[:5, :5])
    print('*' * 64)
    #
    C = np.zeros((n, m))
    start_time = time.perf_counter()
    matmul_CPU(A, B, C)
    end_time = time.perf_counter()
    print('Total Time with Numba CPU: %fs.' % (end_time - start_time))
    print(C[:5, :5])
    print('*' * 64)
    #
    C = np.zeros((n, m))
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    start_time = time.perf_counter()
    matmul_CUDA[n, m](d_A, d_B, d_C)
    end_time = time.perf_counter()
    C = np.empty(shape = d_C.shape, dtype = d_C.dtype)
    d_C.copy_to_host(C)
    print('Total Time with Numba CUDA: %fs.' % (end_time - start_time))
    print(C[:5, :5])
    print('*' * 64)
    #
    tc.init(arch = tc.gpu, default_fp = tc.f64, default_ip = tc.i64)
    C = np.zeros((n, m))
    global A_tc
    global B_tc
    global C_tc
    A_tc = tc.field(tc.f64, shape = (n, d))
    B_tc = tc.field(tc.f64, shape = (d, m))
    C_tc = tc.field(tc.f64, shape = (n, m))
    A_tc.from_numpy(A)
    B_tc.from_numpy(B)
    C_tc.from_numpy(C)
    start_time = time.perf_counter()
    matmul_TaiChi(d)
    end_time = time.perf_counter()
    print('Total Time with TaiChi: %fs.' % (end_time - start_time))
    print(C_tc[4, 4])
    print('*' * 64)

################################################################################
if __name__ == '__main__':
    main()
