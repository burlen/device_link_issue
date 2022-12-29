#ifndef devut_h
#define devut_h

#include "devut_config.h"
#include <iostream>

#if defined(__CUDACC__)

DEVUT_EXPORT
void partition_thread_blocks(size_t array_size,
    int warps_per_block, dim3 &block_grid, int &n_blocks, dim3 &thread_grid);

DEVUT_EXPORT
inline __device__ size_t index()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

#define check_cuda() \
{ \
    cudaError_t ierr = cudaGetLastError(); \
    if (ierr != cudaSuccess) \
    { \
        std::cerr << "ERROR: [" << __FILE__ << ":" << __LINE__ << "]" << std::endl \
            << cudaGetErrorString(ierr) << std::endl; \
    } \
}
#endif

#endif
