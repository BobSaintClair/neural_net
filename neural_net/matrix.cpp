#include "matrix.h"
#include <iostream>

Vector::Vector(const std::vector<double>& data)
	: m_data{ data }
{
}

Vector::Vector(const size_t vec_size)
	: m_data{ std::vector<double>(vec_size) }
{
}

void Vector::print() const
{
	std::cout << "Size:" << '\t' << m_data.size() << '\n';
	std::cout << "Cap:" << '\t' << m_data.capacity() << '\n';
	for (size_t i{ 0 }; i < m_data.size(); i++)
	{
		std::cout << m_data.at(i) << '\n';
	}
	std::cout << '\n';
}

size_t Vector::size() const
{
	return m_data.size();
}

void Vector::clear()
{
	m_data.clear();
}

void Vector::setElement(const double value, const size_t idx)
{
	m_data.at(idx) = value;
}

void Vector::removeElement(const size_t idx)
{
	if (idx >= m_data.size())
		throw std::invalid_argument("Index exceeds dimensions!");
	m_data.erase(m_data.begin() + idx);
}

double Vector::getElement(const size_t idx) const
{
	return m_data.at(idx);
}

double& Vector::at(const size_t idx)
{
	return m_data.at(idx);
}

const std::vector<double>& Vector::getDataAsVector() const
{
	return m_data;
}

double Vector::length() const
{
	if (m_data.size() == 0)
		throw std::invalid_argument("Vector is empty!");
	double sum_of_squares{ 0.0 };
	for (size_t i{ 0 }; i < m_data.size(); i++)
	{
		sum_of_squares += m_data.at(i) * m_data.at(i);
	}
	return std::sqrt(sum_of_squares);
}

double Vector::dotProduct(const Vector& other_vector) const
{
	if (m_data.size() != other_vector.m_data.size())
		throw std::invalid_argument("Vector don't have the same dimension!");
	else if (m_data.size() == 0)
		throw std::invalid_argument("Vectors are empty!");

	double sum{ 0.0 };
	for (size_t i{ 0 }; i < m_data.size(); i++)
	{
		sum += m_data.at(i) * other_vector.m_data.at(i);
	}
	return sum;
}

double Vector::distanceTo(const Vector& other_vector) const
{
	if (m_data.size() != other_vector.m_data.size())
		throw std::invalid_argument("Vector don't have the same dimension!");
	else if (m_data.size() == 0)
		throw std::invalid_argument("Vectors are empty!");

	double sum_of_squares{ 0.0 };
	for (size_t i{ 0 }; i < m_data.size(); i++)
	{
		sum_of_squares += (m_data.at(i) - other_vector.m_data.at(i)) * (m_data.at(i) - other_vector.m_data.at(i));
	}
	return std::sqrt(sum_of_squares);
}

void Vector::operator+=(const Vector& other_vector)
{
	if (m_data.size() != other_vector.m_data.size())
		throw std::invalid_argument("Vector don't have the same dimension!");

	for (size_t i{ 0 }; i < m_data.size(); i++)
	{
		m_data.at(i) += other_vector.m_data.at(i);
	}
}

void Vector::operator-=(const Vector& other_vector)
{
	if (m_data.size() != other_vector.m_data.size())
		throw std::invalid_argument("Vector don't have the same dimension!");

	for (size_t i{ 0 }; i < m_data.size(); i++)
	{
		m_data.at(i) -= other_vector.m_data.at(i);
	}
}

void Vector::operator*=(const double multiplier)
{
	for (size_t i{ 0 }; i < m_data.size(); i++)
	{
		m_data.at(i) *= multiplier;
	}
}

Vector Vector::operator+(const Vector& other_vector) const
{
	Vector result{ *this };
	result += other_vector;
	return result;
}

Vector Vector::operator-(const Vector& other_vector) const
{
	Vector result{ *this };
	result -= other_vector;
	return result;
}

Vector Vector::operator*(const double multiplier) const
{
	Vector result{ *this };
	result *= multiplier;
	return result;
}

double& Vector::operator[](const size_t idx)
{
	return m_data.at(idx);
}

Matrix::Matrix(const size_t nrow, const size_t ncol, const std::vector<double>& data) //constructor, called when an object is created, don't include default vars here
    : m_nrow{ nrow }, m_ncol{ ncol }, m_data{ data }
{
    if ((nrow * ncol) != data.size())
        throw std::invalid_argument("Data size does not match the dimensions!");
    if ((nrow == 0 && ncol > 0) || (nrow > 0 && ncol == 0))
        throw std::invalid_argument("One dimension is zero while the other one is not!");
}

void Matrix::print() const
{
    std::cout << "Size:" << '\t' << m_data.size() << '\n';
    std::cout << "Cap:" << '\t' << m_data.capacity() << '\n';
    std::cout << "nRow:" << '\t' << m_nrow << '\n';
    std::cout << "nCol:" << '\t' << m_ncol << '\n';
    size_t idx{ 0 };
    for (size_t i{ 0 }; i < m_nrow; i++)
    {
        for (size_t j{ 0 }; j < m_ncol; j++)
        {
            std::cout << m_data.at(idx) << ' ';
            idx++;
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

size_t Matrix::nRow() const
{
    return m_nrow;
}

size_t Matrix::nCol() const
{
    return m_ncol;
}

void Matrix::clear()
{
    m_nrow = 0;
    m_ncol = 0;
    m_data.clear();
}

void Matrix::setElement(const double value, const size_t row_idx, const size_t col_idx)
{
    if (row_idx >= m_nrow || col_idx >= m_ncol)
        throw std::invalid_argument("Index exceeds dimensions!");
    m_data.at(row_idx * m_ncol + col_idx) = value;
}

void Matrix::setElement(const double value, const size_t idx)
{
    m_data.at(idx) = value;
}

void Matrix::removeRow(const size_t row_idx)
{
    if (row_idx >= m_nrow)
        throw std::invalid_argument("Index exceeds dimensions!");
    else if (m_nrow == 1)
    {
        m_nrow = 0;
        m_ncol = 0;
        m_data.clear();
    }
    else
    {
        size_t idx_start{ row_idx * m_ncol };
        m_data.erase(std::next(m_data.begin(), idx_start), std::next(m_data.begin(), idx_start + m_ncol));
        m_nrow--;

        if ((m_nrow * m_ncol) != m_data.size())
            throw std::invalid_argument("Data size does not match the dimensions!");
    }
}

void Matrix::removeCol(const size_t col_idx)
{
    if (col_idx >= m_ncol)
        throw std::invalid_argument("Index exceeds dimensions!");
    else if (m_ncol == 1)
    {
        m_nrow = 0;
        m_ncol = 0;
        m_data.clear();
    }
    else
    {
        size_t idx_from_back{ (m_nrow - 1) * m_ncol + col_idx };

        for (size_t i{ 0 }; i < m_nrow; i++)
        {
            m_data.erase(m_data.begin() + idx_from_back);
            idx_from_back -= m_ncol;
        }
        m_ncol--;

        if ((m_nrow * m_ncol) != m_data.size())
            throw std::invalid_argument("Data size does not match the dimensions!");
    }
}

double Matrix::getElement(const size_t row_idx, const size_t col_idx) const
{
    if (row_idx >= m_nrow || col_idx >= m_ncol)
        throw std::invalid_argument("Index exceeds dimensions!");
    return m_data.at(row_idx * m_ncol + col_idx);
}

double Matrix::getElement(const size_t idx) const
{
    return m_data.at(idx);
}

double& Matrix::at(const size_t row_idx, const size_t col_idx)
{
    if (row_idx >= m_nrow || col_idx >= m_ncol)
        throw std::invalid_argument("Index exceeds dimensions!");
    return m_data.at(row_idx * m_ncol + col_idx);
}

double& Matrix::at(const size_t idx)
{
    return m_data.at(idx);
}

const std::vector<double>& Matrix::getDataAsVector() const
{
    return m_data;
}

bool Matrix::isSquare() const
{
    return (m_nrow == m_ncol);
}

void Matrix::operator+=(const Matrix& other_matrix)
{
    if (m_ncol != other_matrix.m_ncol || m_nrow != other_matrix.m_nrow)
        throw std::invalid_argument("Matrices have different dimensions!");

    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        m_data.at(i) += other_matrix.m_data.at(i);
    }
}

void Matrix::operator-=(const Matrix& other_matrix)
{
    if (m_ncol != other_matrix.m_ncol || m_nrow != other_matrix.m_nrow)
        throw std::invalid_argument("Matrices have different dimensions!");

    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        m_data.at(i) -= other_matrix.m_data.at(i);
    }
}

void Matrix::operator*=(const double multiplier)
{
    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        m_data.at(i) *= multiplier;
    }
}

Matrix Matrix::operator+(const Matrix& other_matrix) const
{
    Matrix result{ *this };
    result += other_matrix;
    return result;
}

Matrix Matrix::operator-(const Matrix& other_matrix) const
{
    Matrix result{ *this };
    result -= other_matrix;
    return result;
}

Matrix Matrix::operator*(const double multiplier) const
{
    Matrix result{ *this };
    result *= multiplier;
    return result;
}

Matrix Matrix::operator*(const Matrix& other_matrix) const
{
    if (m_ncol != other_matrix.m_nrow)
        throw std::invalid_argument("Matrices are not compatible for multiplication!");

    size_t sum_over{ m_ncol };
    size_t nrow{ m_nrow };
    size_t ncol{ other_matrix.m_ncol };

    Matrix result{ nrow, ncol, std::vector<double>(nrow * ncol) };

    for (size_t i{ 0 }; i < nrow; i++)
    {
        for (size_t j{ 0 }; j < ncol; j++)
        {
            double element_value{ 0.0 };

            for (size_t k{ 0 }; k < sum_over; k++)
            {
                element_value += m_data.at(i * m_ncol + k) * other_matrix.m_data.at(k * other_matrix.m_ncol + j);
            }

            result.m_data.at(i * ncol + j) = element_value;
        }
    }

    return result;
}

double& Matrix::operator[](const size_t idx)
{
    return m_data.at(idx);
}

Vector operator*(const Matrix& matrix1, const Vector& vector1)
{
    if (matrix1.nCol() != vector1.size())
        throw std::invalid_argument("Matrices are not compatible for multiplication!");

    size_t sum_over{ matrix1.nCol() };
    size_t nrow{ matrix1.nRow() };

    std::vector<double> result(nrow);

    for (size_t i{ 0 }; i < nrow; i++)
    {
        double element_value{ 0.0 };

        for (size_t k{ 0 }; k < sum_over; k++)
        {
            element_value += matrix1.getElement(i, k) * vector1.getElement(k);
        }

        result.at(i) = element_value;
    }

    return Vector{ result };
}