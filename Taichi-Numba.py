import numba as nb
from numba import cuda
import numpy as np
import taichi as tc
import time
import matplotlib.pyplot as plt
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
def main0(A, B, n, d, m, times):
    #
    C = np.zeros((n, m))
    start_time = time.perf_counter()
    C = np.dot(A, B)
    end_time = time.perf_counter()
    times[0].append(end_time - start_time)
    print('Total Time with Numpy dot: %fs.' % (times[0][-1]))
    #print(C[:5, :5])
    print('*' * 64)
    #
    '''
    C = np.zeros((n, m))
    start_time = time.perf_counter()
    matmul_CPU(A, B, C)
    end_time = time.perf_counter()
    times[1].append(end_time - start_time)
    print('Total Time with Numba CPU: %fs.' % (times[1][-1]))
    #print(C[:5, :5])
    print('*' * 64)
    '''
    #
    C = np.zeros((n, m))
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    start_time = time.perf_counter()
    matmul_CUDA[n, m](d_A, d_B, d_C)
    end_time = time.perf_counter()
    #C = np.empty(shape = d_C.shape, dtype = d_C.dtype)
    #d_C.copy_to_host(C)
    C = d_C.copy_to_host()
    times[2].append(end_time - start_time)
    print('Total Time with Numba CUDA: %fs.' % (times[2][-1]))
    #print(C[:5, :5])
    print('*' * 64)

################################################################################
def main1(A, B, n, d, m, times):
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
    times[3].append(end_time - start_time)
    print('Total Time with TaiChi: %fs.' % (times[3][-1]))
    #print(C_tc[4, 4])
    print('*' * 64)

################################################################################
if __name__ == '__main__':
    m = 1024
    nd = [x for x in range(512, 7168, 512)]
    As = [np.random.rand(n, n) for n in nd]
    Bs = [np.random.rand(n, m) for n in nd]
    times = [[] for x in range(4)]
    for i in range(len(nd)):
        main0(As[i], Bs[i], nd[i], nd[i], m, times)
    for i in range(len(nd)):
        main1(As[i], Bs[i], nd[i], nd[i], m, times)
    #
    fig, ax = plt.subplots()
    ax.plot(nd, times[0], 'o-', color = 'b', label='Numpy dot')
    #ax.plot(nd, times[1], 'o-', color = 'g', label='Numba CPU')
    ax.plot(nd, times[2], 'o-', color = 'r', label='Numba CUDA')
    ax.plot(nd, times[3], 'o-', color = 'y', label='Taichi')
    ax.set(xlabel = 'n', ylabel = 'time')
    ax.legend(loc = 'best', shadow = True, fontsize = 'x-large')
    ax.grid()
    fig.savefig("test.png")
    plt.show()
