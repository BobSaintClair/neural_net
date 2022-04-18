#include "read_csv.h"
#include <fstream>
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <iostream>
#include <iomanip> // for output manipulator std::setprecision()

data_frame read_csv(std::string filename)
{
    std::ifstream myFile{ filename };
    std::vector<std::string> colnames{};
    std::vector<double> data{};

    if (!myFile.is_open())
        throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line{ "" };
    std::string colname{ "" };
    double val{ 0.0 };

    size_t n_cols{ 0 };
    size_t n_rows{ 0 };

    if (myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);

        // Create a stringstream from line
        std::stringstream ss{ line };

        // Extract each column name
        while (std::getline(ss, colname, ','))
        {
            n_cols++;
            colnames.push_back(colname);
        }
    }

    // Read data, line by line
    while (std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss{ line };
        n_rows++;
        // Keep track of the current column index
        while (ss >> val)
        {
            // Add the current integer to the 'colIdx' column's values vector
            data.push_back(val);
            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',') 
                ss.ignore();
        }
    }

    return std::pair{ colnames, Matrix{ n_rows, n_cols, data } };
}

void print_data_frame(data_frame print_me, std::ostream& stream)
{
    stream << std::fixed;
    stream << std::setprecision(3);
    stream << "Rows: " << print_me.second.nRow() << '\n';
    stream << "Cols: " << print_me.second.nCol() << '\n';
    for (size_t i{ 0 }; i < print_me.first.size(); i++)
    {
        stream << print_me.first[i] << '\t';
    }
    stream << '\n';
    for (size_t i{ 0 }; i < std::min(5, static_cast<int>(print_me.second.nRow())); i++)
    {
        for (size_t j{ 0 }; j < print_me.second.nCol(); j++)
        {
            stream << print_me.second.at(i, j) << '\t';
        }
        stream << '\n';
    }
    stream << "Top 5 rows printed." << '\n';
    stream << '\n';
}