#include "bwt.h"
#include <assert.h>

void Test(std::string input, std::string assertoutput, std::string resource){
    std::cout << std::endl;
    std::cout << "Source: " <<  resource << std::endl;
    std::cout << "Testing: " << input << std::endl;
    auto result = BWT(input);
    std::cout << "Result:  ";
    std::string output;
    for (auto c : result.data ){
        output.push_back(c);
        std::cout << c;
    }
    std::cout << std::endl;
    assert(output == assertoutput);
}

int main(){
    Test("^BANANA|", "BNN^AA|A", "https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform");
    Test("banana$", "annb$aa", "https://www.geeksforgeeks.org/burrows-wheeler-data-transform-algorithm/");
    Test("SIX.MIXED.PIXIES.SIFT.SIXTY.PIXIE.DUST.BOXES", "TEXYDST.E.IXIXIXXSSMPPS.B..E.S.EUSFXDIIOIIIT", "https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform");
}