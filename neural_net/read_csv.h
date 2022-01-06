#pragma once
#include <string>
#include <utility> // std::pair
#include <vector>
#include "matrix.h"

using data_frame = std::vector<std::pair<std::string, std::vector<double>>>;
using data_frame_unlabeled = std::pair<Vector, Matrix>;
constexpr int n_rows_to_print{ 6 };

data_frame read_csv1(std::string filename);
data_frame_unlabeled read_csv2(std::string filename);
void print_data_frame(data_frame print_me);