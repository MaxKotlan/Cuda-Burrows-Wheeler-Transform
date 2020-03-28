#ifndef BWT_H
#define BWT_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <time.h>

struct TransformedData{
    unsigned int originalIndex;
    std::vector<unsigned char> data;

    /*Move operator to only shallow copy data*/
    TransformedData& operator=(const TransformedData& other) {
        originalIndex = other.originalIndex;
        data = std::move(other.data);
        return *this;
    }
};
#include "bwt_cpu.h"
#include "bwt_cuda.h"

#endif