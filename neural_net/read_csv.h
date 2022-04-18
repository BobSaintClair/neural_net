#pragma once
#include <string>
#include <utility> // std::pair
#include <vector>
#include "matrix.h"

using data_frame = std::pair<std::vector<std::string>, Matrix>;

data_frame read_csv(std::string filename);
void print_data_frame(data_frame print_me, std::ostream& stream = std::cout);