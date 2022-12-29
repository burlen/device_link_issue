#include "div.h"
#include "add.h"

/*
#include "devut.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__
void div_kernel(const double *a , const double *b, double *c, size_t n)
{
    size_t i = index();

    if (i >= n) return;

    c[i] = a[i] / b[i];
}


void div(const double *a , const double *b, double *c, size_t n)
{
    double *deva, *devb, *devc;

    cudaMalloc(&deva, n*sizeof(double));
    cudaMalloc(&devb, n*sizeof(double));
    cudaMalloc(&devc, n*sizeof(double));

    cudaMemcpy(deva, a, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devb, b, n*sizeof(double), cudaMemcpyHostToDevice);

    size_t nblks = n / 32;
    nblks += n % 32 ? 1 : 0;

    div_kernel<<<32, nblks>>>(deva, devb, devc, n);

    cudaMemcpy(c, devc, n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
}
*/

void div_cpu(const double *a , const double *b, double *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] / b[i];
}

void add_div(const double *a , const double *b, const double *c, double *d, size_t n)
{
    div_cpu(a,b,d,n);
    add(d,c,d,n);
}


