#include "bwt.h"
#include <assert.h>

void Test(std::string input, std::string assertoutput){
    std::cout << "Testing: " << input << std::endl;
    auto result = BWT::BWT(input);
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
    Test("^BANANA@", "BNN^AA@A");
    Test("banana@", "annb@aa");
    Test("SIX.MIXED.PIXIES.SIFT.SIXTY.PIXIE.DUST.BOXES", "TEXYDST.E.IXIXIXXSSMPPS.B..E.S.EUSFXDIIOIIIT");
}