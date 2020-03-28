#include "bwt_cpu.h"
#include <vector>
#include <algorithm>
#include <iostream>

TransformedData BWT(const std::vector<unsigned char>& data){
    unsigned int k = data.size();
    std::vector<unsigned int> sortedindex(k);
    std::vector<unsigned char> transformeddata(k);
    TransformedData result;
    result.data = std::move(transformeddata);

    for (unsigned int i = 0; i < k; i++)
        sortedindex[i] = i;

    std::sort(sortedindex.begin(), sortedindex.end(), [&](unsigned int a, unsigned int b){
        unsigned int resa = 0; unsigned int resb = 0;
        for (int i = 0; i < k && resa == resb; i++){
            unsigned int la = i-a+k;
            unsigned int lb = i-b+k;
            resa += data[(la)%(k)];
            resb += data[(lb)%(k)];
        }
        return resa < resb;
    });

    for (unsigned int i = 0; i < k; i++){
        unsigned int r = sortedindex[i];
        if (r == 0) result.originalIndex = i;
        result.data[i] = data[(-1 - r + k)%(k)];
    }

    return std::move(result);
}

std::vector<unsigned char> INVERSE_BWT(const TransformedData &transformdata){
    unsigned int k = transformdata.data.size();
    std::vector<std::vector<unsigned char>> original(k);
    for (int i = 0; i < k; i++){
        for (int p = 0; p < k; p++)
            original[p].push_back(transformdata.data[p]);
        
        std::sort(original.begin(), original.end(), [&](std::vector<unsigned char>& a, std::vector<unsigned char>& b){
            unsigned int resa = 0; unsigned int resb = 0;
            for (int z = 0; z < i+1 && resa == resb; z++){
                resa += a[i];
                resb += b[i];
            }
            return resa < resb;
        });
    }

    for (auto v : original){
        for (auto c : v)
            std::cout << c;
        std::cout << std::endl;
    }

    return transformdata.data;
}

TransformedData BWT(std::string input){
    std::vector<unsigned char> data(input.size());
    for (int i = 0; i < input.size(); i++)
        data[i] = input[i];
    return std::move(BWT(data));
}