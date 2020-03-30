#ifndef BWT_CUDA_H
#define BWT_CUDA_H
#include "bwt.h"

TransformedData BWT_CUDA_BITONIC_SORT(const std::vector<unsigned char>& data);

#endif