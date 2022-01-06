#include "read_csv.h"
#include <fstream>
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <iostream>
#include <iomanip> // for output manipulator std::setprecision()

data_frame read_csv1(std::string filename)
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

data_frame_unlabeled read_csv2(std::string filename)
{
    std::vector<double> y{};
    std::vector<double> x{};

    std::ifstream myFile{ filename };

    if (!myFile.is_open())
        throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line{ "" };
    std::string colname{ "" };
    double val{ 0.0 };

    size_t ncol{ 0 };
    if (myFile.good())
    {
        std::getline(myFile, line);
        std::stringstream ss{ line };

        while (std::getline(ss, colname, ','))
        {
            ncol++;
        }
    }

    size_t nrow{ 0 };
    while (std::getline(myFile, line))
    {
        std::stringstream ss{ line };
        int colIdx{ 0 };

        while (ss >> val)
        {
            if (colIdx == 0)
                y.push_back(val);
            else
                x.push_back(val);

            if (ss.peek() == ',') ss.ignore();
            colIdx++;
        }
        nrow++;
    }

    return std::pair{ y, Matrix{ nrow, ncol - 1, x } };
}

void print_data_frame(data_frame print_me)
{
    std::cout << std::setprecision(4);
    std::cout << "Rows: " << print_me.at(0).second.size() << '\n';
    std::cout << "Cols: " << print_me.size() << '\n';
    for (size_t i{ 0 }; i < print_me.size(); i++)
    {
        std::cout << print_me.at(i).first << '\t';
    }
    std::cout << '\n';
    size_t min{ (print_me.at(0).second.size() > n_rows_to_print) ? n_rows_to_print : print_me.at(0).second.size() };
    for (size_t j{ 0 }; j < min; j++)
    {
        for (size_t i{ 0 }; i < print_me.size(); i++)
        {
            std::cout << print_me.at(i).second.at(j) << '\t';
        }
        std::cout << '\n';
    }
    std::cout << "..................................";
    std::cout << '\n';
}