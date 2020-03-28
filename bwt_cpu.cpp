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
    std::vector<std::vector<int>> original(k);
    std::vector<unsigned char> data(k);
    for (int i = 0; i < k; i++){
        for (int p = 0; p < k; p++)
            original[p].push_back(p);
        
        std::sort(original.begin(), original.end(), [&](std::vector<int>& a, std::vector<int>& b){
            unsigned int resa = 0; unsigned int resb = 0;
            for (int z = 0; z < i+1 && resa == resb; z++){
                resa += transformdata.data[a[i-z]];
                resb += transformdata.data[b[i-z]];
            }
            return resa < resb;
        });

        for (int h = 0; h < k; h++) {
            for (int p = 0; p <= i; p++)
                std::cout << original[h][p] << " ";//<< transformdata.data[original[h][p]] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

    }

    for (int j = 0; j < original.size(); j++){
        std::cout << j << ": ";
        for(int c = 0; c < original.size(); c++){
            std::cout << transformdata.data[original[j][c]];
            if (transformdata.originalIndex ==  j)
            data[original.size()-1 - c] = transformdata.data[original[j][c]];
        }
        std::cout << std::endl;
    }

    return data;
}

TransformedData BWT(std::string input){
    std::vector<unsigned char> data(input.size());
    for (int i = 0; i < input.size(); i++)
        data[i] = input[i];
    return std::move(BWT(data));
}