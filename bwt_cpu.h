#ifndef BWT_CPU_H
#define BWT_CPU_H
#include "bwt.h"

TransformedData BWT(const std::vector<unsigned char>& data);
TransformedData BWT(std::string input);
std::vector<unsigned char> INVERSE_BWT(const TransformedData &transformdata);

#endif