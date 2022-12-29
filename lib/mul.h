#ifndef mul_h
#define mul_h

#include "math_config.h"
#include <cstdlib>

/// c = a*b
MATH_EXPORT
void mul(const double *a , const double *b, double *c, size_t n);

/// c = a*b
MATH_EXPORT
void mul_cuda(const double *a , const double *b, double *c, size_t n);

/// c = a*b
MATH_EXPORT
void mul_cpu(const double *a , const double *b, double *c, size_t n);

/// d = a*b + c
MATH_EXPORT
void add_mul(const double *a , const double *b, const double *c, double *d, size_t n);

/// d = a*b + c
MATH_EXPORT
void add_mul_cuda(const double *a , const double *b, const double *c, double *d, size_t n);

#endif
