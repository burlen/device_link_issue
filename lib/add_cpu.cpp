#include <cstdlib>

void add_cpu(const double *a , const double *b, double *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}



