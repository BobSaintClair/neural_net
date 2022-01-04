#include "read_csv.h"
#include <fstream>
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

data_frame read_csv(std::string filename)
{
    data_frame result{};

    std::ifstream myFile{ filename };

    if (!myFile.is_open())
        throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line{ "" };
    std::string colname{ "" };
    double val{ 0.0 };

    // Read the column names
    if (myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);

        // Create a stringstream from line
        std::stringstream ss{ line };

        // Extract each column name
        while (std::getline(ss, colname, ','))
        {
            result.push_back({ colname, std::vector<double> {} });
        }
    }

    // Read data, line by line
    while (std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss{ line };

        // Keep track of the current column index
        int colIdx{ 0 };

        // Extract each integer
        while (ss >> val)
        {
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);
            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',') ss.ignore();

            // Increment the column index
            colIdx++;
        }
    }

    return result;
}