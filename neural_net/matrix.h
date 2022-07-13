#pragma once
#include <vector>
#include <iostream>

std::vector<size_t> sliceVector(const std::vector<size_t>& vector, const size_t first_idx, const size_t second_idx);

class Matrix
{
private:
    size_t m_nrow{};
    size_t m_ncol{};
    std::vector<double> m_data{};

public:
    Matrix(const size_t nrow = 0, const size_t ncol = 0, const std::vector<double>& data = std::vector<double>{});

    void print(std::ostream& stream = std::cout) const;
    void printDims(std::ostream& stream = std::cout) const;
    size_t size() const;
    size_t nRow() const;
    size_t nCol() const;
    void clear();
    double sumElements() const;
    double dotProduct(const Matrix& other_matrix) const;
    void removeRow(const size_t row_idx);
    void removeCol(const size_t col_idx);
    Matrix getRow(const size_t row_idx) const;
    Matrix getRows(const std::vector<size_t> row_idx) const;
    Matrix getCol(const size_t col_idx) const;
    Matrix getCols(const std::vector<size_t> col_idx) const;
    double& at(const size_t row_idx, const size_t col_idx);
    double& at(const size_t idx);
    const double& at(const size_t row_idx, const size_t col_idx) const;
    const double& at(const size_t idx) const;
    bool isSquare() const;
    Matrix transpose() const;
    void transposeMe();
    void zeroMe();
    Matrix hadamardProduct(const Matrix& other_matrix) const;
    Matrix hadamardProductColumnwise(const Matrix& other_matrix) const;
    Matrix zeroButOne(const size_t idx) const;
    Matrix zeroButOne(const size_t row_idx, const size_t col_idx) const;
    Matrix zeroButOneRow(const size_t row_idx) const;
    std::vector<double> columnwiseMean() const;
    std::vector<double> columnwiseStdDev() const;

    void operator+=(const Matrix& other_matrix);
    void operator-=(const Matrix& other_matrix);
    void operator*=(const double multiplier);
    void operator+=(const double add_me);
    void operator-=(const double subtract_me);
    Matrix operator+(const double add_me) const;
    Matrix operator+(const Matrix& other_matrix) const;
    Matrix operator-(const double subtract_me) const;
    Matrix operator-(const Matrix& other_matrix) const;
    Matrix operator*(const double multiplier) const;
    Matrix operator*(const Matrix& other_matrix) const;
    Matrix addColumnwise(const Matrix& other_matrix) const;
    double& operator[](const size_t idx);
    const double& operator[](const size_t idx) const;
    double& operator()(const size_t row_idx, const size_t col_idx);
    const double& operator()(const size_t row_idx, const size_t col_idx) const;
};