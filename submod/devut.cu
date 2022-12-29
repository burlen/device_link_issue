#include "devut.h"

#include <cuda.h>
#include <cuda_runtime.h>

void partition_thread_blocks(size_t array_size,
    int warps_per_block, dim3 &block_grid, int &n_blocks, dim3 &thread_grid)
{
    unsigned long warp_size = 32;
    unsigned long threads_per_block = warps_per_block * warp_size;

    thread_grid.x = threads_per_block;
    thread_grid.y = 1;
    thread_grid.z = 1;

    unsigned long block_size = threads_per_block;
    n_blocks = array_size / block_size;

    if (array_size % block_size)
        ++n_blocks;

    block_grid.x = n_blocks;
    block_grid.y = 1;
    block_grid.z = 1;
}
