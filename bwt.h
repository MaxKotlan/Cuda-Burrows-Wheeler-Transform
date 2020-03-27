#ifndef BWT_H
#define BWT_H

#include <vector>
#include <algorithm>

namespace BWT{

struct TransformedData{
    int originalIndex;
    std::vector<unsigned char> data;
};

TransformedData BWT(const std::vector<unsigned char>& data){
    int k = data.size();
    std::vector<int> sortedindex(k);
    std::vector<unsigned char> transformeddata(k);

    for (int i = 0; i < k; i++)
        sortedindex[i] = i;

    std::sort(sortedindex.begin(), sortedindex.end(), [&](int a, int b){
        unsigned int resa = 0; unsigned int resb = 0;
        for (int i = 0; i < k && resa == resb; i++){
            unsigned int la = i-a+k;
            unsigned int lb = i-b+k;
            resa += data[(la)%(k)]+ (k*8*sizeof(unsigned char) - 8*i*sizeof(unsigned char));
            resb += data[(lb)%(k)]+ (k*8*sizeof(unsigned char) - 8*i*sizeof(unsigned char));
        }
        return resa < resb;
    });

    int originalIndex = 0;
    for (int i = 0; i < data.size(); i++){
        int r = sortedindex[i];
        if (r == 0) originalIndex = i;
        transformeddata[i] = data[(-1 - r + data.size())%(data.size())];
    }
    return {originalIndex, std::move(transformeddata)};
}

}

#endif