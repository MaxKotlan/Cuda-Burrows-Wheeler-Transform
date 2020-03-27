#include <iostream>
#include <vector>
#include <algorithm>
#include "bwt.h"

int main(int argc, char** argv){
    std::vector<unsigned char> data;
    for (auto c : "SIX.MIXED.PIXIES.SIFT.SIXTY.PIXIE.DUST.BOXES")
        if (c != '\0')
            data.push_back(c);

    BWT::TransformedData t = BWT::BWT(data);
    std::cout << t.originalIndex << std::endl;
    for (auto c : t.data)
        std::cout << c;

}