#pragma once
#include <vector>
#include <iostream>

class Vector
{
private:
    std::vector<double> m_data{};

public:
    Vector(const std::vector<double>& data = std::vector<double>{});
    Vector(const size_t vec_size);
    Vector(const size_t vec_size, const double value);

    void print(std::ostream& stream = std::cout) const;
    size_t size() const;
    void clear();
    void pushBack(const double value);
    void removeElement(const size_t idx);
    double& at(const size_t idx);
    const double& at(const size_t idx) const;
    double length() const;
    double dotProduct(const Vector& other_vector) const;
    double distanceTo(const Vector& other_vector) const;

    void operator+=(const Vector& other_vector);
    void operator-=(const Vector& other_vector);
    void operator*=(const double multiplier);
    Vector operator+(const Vector& other_vector) const;
    Vector operator-(const Vector& other_vector) const;
    Vector operator*(const double multiplier) const;
    double& operator[](const size_t idx);
    const double& operator[](const size_t idx) const;
};

class Matrix
{
private:
    size_t m_nrow{};
    size_t m_ncol{};
    std::vector<double> m_data{};

public:
    Matrix(const size_t nrow = 0, const size_t ncol = 0, const std::vector<double>& data = std::vector<double>{});

    void print(std::ostream& stream = std::cout) const;
    size_t nRow() const;
    size_t nCol() const;
    void clear();
    void removeRow(const size_t row_idx);
    void removeCol(const size_t col_idx);
    Vector getRow(const size_t row_idx) const;
    Vector getCol(const size_t col_idx) const;
    double& at(const size_t row_idx, const size_t col_idx);
    double& at(const size_t idx);
    const double& at(const size_t row_idx, const size_t col_idx) const;
    const double& at(const size_t idx) const;
    bool isSquare() const;

    void operator+=(const Matrix& other_matrix);
    void operator-=(const Matrix& other_matrix);
    void operator*=(const double multiplier);
    Matrix operator+(const Matrix& other_matrix) const;
    Matrix operator-(const Matrix& other_matrix) const;
    Matrix operator*(const double multiplier) const;
    Matrix operator*(const Matrix& other_matrix) const;
    double& operator[](const size_t idx);
    const double& operator[](const size_t idx) const;
};

Vector operator*(const Matrix& matrix1, const Vector& vector1);