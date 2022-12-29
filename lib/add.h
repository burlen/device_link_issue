#ifndef add_h
#define add_h

#include "math_config.h"
#include <cstdlib>

#if defined(__CUDACC__)
MATH_EXPORT
__global__
void add_kernel(const double *a , const double *b, double *c, size_t n);
#endif

MATH_EXPORT
void add(const double *a , const double *b, double *c, size_t n);

MATH_EXPORT
void add_cpu(const double *a , const double *b, double *c, size_t n);

#endif
