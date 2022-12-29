#include "devut.h"
#include "add.h"
#include "mul.h"
#include "div.h"

#include <vector>
#include <iostream>
#include <cstdlib>


#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void div_kernel_2(const double *a , const double *b, double *c, size_t n)
{
    size_t i = index();

    if (i >= n) return;

    c[i] = a[i] / b[i];
}


void div_2(const double *a , const double *b, double *c, size_t n)
{
    double *deva, *devb, *devc;

    cudaMalloc(&deva, n*sizeof(double));
    cudaMalloc(&devb, n*sizeof(double));
    cudaMalloc(&devc, n*sizeof(double));

    cudaMemcpy(deva, a, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devb, b, n*sizeof(double), cudaMemcpyHostToDevice);

    size_t nblks = n / 32;
    nblks += n % 32 ? 1 : 0;

    div_kernel_2<<<32, nblks>>>(deva, devb, devc, n);

    cudaMemcpy(c, devc, n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
}
#endif





int main(int, char **)
{
  size_t n = 32;

  std::vector<double> a(n, 2.0);
  std::vector<double> b(n, 2.0);
  std::vector<double> c(n, 0.0);

  add(a.data(), b.data(), c.data(), n);
  mul(c.data(), b.data(), c.data(), n);

#if defined(__CUDACC__)
  div_2(a.data(), c.data(), c.data(), n);
#endif

  std::cerr << "c = ";
  for (size_t i =0; i < n; ++i)
      std::cerr << c[i] << ", ";
  std::cerr << std::endl;

  return 0;
}

