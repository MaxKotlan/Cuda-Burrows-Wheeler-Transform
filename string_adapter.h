#ifndef STRING_ADAPTER_H
#define STRING_ADAPTER_H

#include <vector>
#include <string>

std::vector<unsigned char> toByteArray(const std::string& input);
std::string toString(const std::vector<unsigned char>& input);

#endif