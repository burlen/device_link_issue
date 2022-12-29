#ifndef div_h
#define div_h

#include "math_config.h"
#include <cstdlib>

/// c = a/b
MATH_EXPORT
void div_cpu(const double *a , const double *b, double *c, size_t n);

/// d = a/b + c
MATH_EXPORT
void add_div(const double *a , const double *b, const double *c, double *d, size_t n);

#endif
