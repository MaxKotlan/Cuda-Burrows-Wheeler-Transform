#ifndef BWT_H
#define BWT_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <time.h>

namespace BWT{

struct TransformedData{
    int originalIndex;
    std::vector<unsigned char> data;

    /*Move operator to only shallow copy data*/
    TransformedData& operator=(const TransformedData& other) {
        originalIndex = other.originalIndex;
        data = std::move(other.data);
        return *this;
    }
};

TransformedData BWT(const std::vector<unsigned char>& data){
    int k = data.size();
    std::vector<int> sortedindex(k);
    std::vector<unsigned char> transformeddata(k);
    TransformedData result;
    result.data = std::move(transformeddata);

    for (int i = 0; i < k; i++)
        sortedindex[i] = i;

    std::sort(sortedindex.begin(), sortedindex.end(), [&](int a, int b){
        unsigned int resa = 0; unsigned int resb = 0;
        for (int i = 0; i < k && resa == resb; i++){
            unsigned int la = i-a+k;
            unsigned int lb = i-b+k;
            resa += data[(la)%(k)];
            resb += data[(lb)%(k)];
        }
        return resa < resb;
    });

    for (int i = 0; i < k; i++){
        int r = sortedindex[i];
        if (r == 0) result.originalIndex = i;
        result.data[i] = data[(-1 - r + k)%(k)];
    }

    return std::move(result);
}

std::vector<unsigned char> INVERSE_BWT(const TransformedData &transformdata){
    return std::vector<unsigned char>(0);
}

TransformedData BWT(std::string input){
    std::vector<unsigned char> data(input.size());
    for (int i = 0; i < input.size(); i++)
        data[i] = input[i];
    return std::move(BWT(data));
}

}

#endif