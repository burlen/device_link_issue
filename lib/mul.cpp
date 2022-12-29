#include "devut.h"
#include "mul.h"
#include "add.h"

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void mul_kernel(const double *a , const double *b, double *c, size_t n)
{
    size_t i = index();

    if (i >= n) return;

    c[i] = a[i] * b[i];
}


void mul_cuda(const double *a , const double *b, double *c, size_t n)
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

    mul_kernel<<<tg,bg>>>(deva, devb, devc, n);

    cudaMemcpy(c, devc, n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
    check_cuda()
}

void add_mul_cuda(const double *a , const double *b, const double *c, double *d, size_t n)
{
    check_cuda()
    double *deva, *devb, *devc, *devd;

    cudaMalloc(&deva, n*sizeof(double));
    cudaMalloc(&devb, n*sizeof(double));
    cudaMalloc(&devc, n*sizeof(double));
    cudaMalloc(&devd, n*sizeof(double));

    cudaMemcpy(deva, a, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devb, b, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devc, c, n*sizeof(double), cudaMemcpyHostToDevice);

    dim3 tg;
    dim3 bg;
    int nb = 0;
    partition_thread_blocks(n, 2, bg, nb, tg);

    mul_kernel<<<tg,bg>>>(deva, devb, devd, n);
    add_kernel<<<tg,bg>>>(devd, devc, devd, n);

    cudaMemcpy(d, devd, n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
    cudaFree(devd);
    check_cuda()
}

void mul(const double *a , const double *b, double *c, size_t n)
{
    mul_cuda(a, b, c, n);
}
#else
void mul(const double *a , const double *b, double *c, size_t n)
{
    mul_cpu(a, b, c, n);
}
#endif

void add_mul(const double *a , const double *b, const double *c, double *d, size_t n)
{
    mul(a,b,d,n);
    add(d,c,d,n);
}
