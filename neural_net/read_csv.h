#pragma once
#include <string>
#include <utility> // std::pair
#include <vector>
using data_frame = std::vector<std::pair<std::string, std::vector<double>>>;
data_frame read_csv(std::string filename);