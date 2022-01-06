#pragma once
#include <string>
#include <utility> // std::pair
#include <vector>
#include "matrix.h"

using data_frame = std::vector<std::pair<std::string, Vector>>;

constexpr int n_rows_to_print{ 6 };

data_frame read_csv(std::string filename);
void print_data_frame(data_frame print_me);