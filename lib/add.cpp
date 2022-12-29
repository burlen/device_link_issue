#include "devut.h"
#include "add.h"

#include <cuda.h>
#include <cuda_runtime.h>


__global__
void add_kernel(const double *a , const double *b, double *c, size_t n)
{
    size_t i = index();

    if (i >= n) return;

    c[i] = a[i] + b[i];
}


void add(const double *a , const double *b, double *c, size_t n)
{
    check_cuda()
    double *deva, *devb, *devc;

    cudaMalloc(&deva, n*sizeof(double));
    cudaMalloc(&devb, n*sizeof(double));
    cudaMalloc(&devc, n*sizeof(double));

    cudaMemcpy(deva, a, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devb, b, n*sizeof(double), cudaMemcpyHostToDevice);


    dim3 tg;
    dim3 bg;
    int nb = 0;
    partition_thread_blocks(n, 2, bg, nb, tg);

    add_kernel<<<tg,bg>>>(deva, devb, devc, n);

    cudaMemcpy(c, devc, n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
    check_cuda()
}



