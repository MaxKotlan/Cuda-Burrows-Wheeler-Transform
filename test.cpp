#include "bwt.h"
#include "bwt_debug.h"
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

void testRandom(unsigned int size){
    std::vector<unsigned char> data(size);
    std::generate_n(data.begin(), size, [](){return rand()%4+'A';});
    std::cout << std::endl;
    PrintUnsortedMatrix(data);
    std::cout << std::endl;
    PrintSortedMatrix(data);
}

int main(){
    Test("123456@", "@123456", "My brain");
    Test("^BANANA|", "BNN^AA|A", "https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform");
    Test("banana$", "annb$aa", "https://www.geeksforgeeks.org/burrows-wheeler-data-transform-algorithm/");
    Test("SIX.MIXED.PIXIES.SIFT.SIXTY.PIXIE.DUST.BOXES", "TEXYDST.E.IXIXIXXSSMPPS.B..E.S.EUSFXDIIOIIIT", "https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform");

    std::cout << std::endl << "Source: https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform:" << std::endl;
    std::cout << "Unsorted Matrix: " << std::endl;;
    PrintUnsortedMatrix("^BANANA|");
    std::cout << std::endl << "Sorted Matrix: " << std::endl;;
    PrintSortedMatrix("^BANANA|");
    testRandom(10);
    //PrintUnsortedMatrix("BANANA");
    //PrintSortedMatrix("BANANA");
    std::cout << BWT("^BANANA|").originalIndex << std::endl;
    auto k = INVERSE_BWT(BWT("SIX.MIXED.PIXIES.SIFT.SIXTY.PIXIE.DUST.BOXES"));
    for (auto c : k)
        std::cout << c;

}