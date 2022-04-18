#pragma once
#include <vector>
#include <iostream>

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
    void removeRow(const size_t row_idx);
    void removeCol(const size_t col_idx);
    Matrix getRow(const size_t row_idx) const;
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
    Matrix zeroButOne(const size_t idx) const;
    Matrix zeroButOne(const size_t row_idx, const size_t col_idx) const;

    void operator+=(const Matrix& other_matrix);
    void operator-=(const Matrix& other_matrix);
    void operator*=(const double multiplier);
    Matrix operator+(const Matrix& other_matrix) const;
    Matrix operator-(const Matrix& other_matrix) const;
    Matrix operator*(const double multiplier) const;
    Matrix operator*(const Matrix& other_matrix) const;
    double& operator[](const size_t idx);
    const double& operator[](const size_t idx) const;
    double& operator()(const size_t row_idx, const size_t col_idx);
    const double& operator()(const size_t row_idx, const size_t col_idx) const;
};