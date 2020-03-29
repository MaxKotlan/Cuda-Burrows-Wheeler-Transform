#include "bwt_cpu.h"
#include <vector>
#include <algorithm>
#include <iostream>


/*
Space Complexity O(3N)
Computation Complexity ? (less than o(n^2) on average)
When testing with LOTR.txt, N = 3311317 and the number of primative operations
was ~509734982 which is only ~153 times large than n. Although it was ~7 time larger than log(3311317) [base 2]
When testing with data that was all the same, it always resulted in O(n^2). 
So It is somewhere between O(nlog(n)) and O(n^2) and largely depends on the data. Not the most efficent probably.
Did not use suffix arrays at all.
*/
TransformedData BWT(const std::vector<unsigned char>& data){
    unsigned int k = data.size();
    std::vector<unsigned int> sortedindex(k);
    std::vector<unsigned char> transformeddata(k);
    TransformedData result;
    result.data = std::move(transformeddata);

    /*vector of row indices*/
    for (unsigned int i = 0; i < k; i++)
        sortedindex[i] = i;

    /*sort row indicies by 'row' (inplace row calculation) lexicographically*/
    std::sort(sortedindex.begin(), sortedindex.end(), [&](unsigned int a, unsigned int b){
        unsigned char diffa = 0; unsigned char diffb = 0;
        for (int i = 0; i < k && diffa == diffb; i++){
            unsigned int la = i-a+k;
            unsigned int lb = i-b+k;
            diffa = data[(la)%(k)];
            diffb = data[(lb)%(k)];
        }
        return diffa < diffb;
    });

    for (auto c: sortedindex)
        std::cout << c << " ";
    std::cout << std::endl;

    /*Find Original Index and Copy to Output Buffer*/
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