#include "matrix.h"
#include <algorithm>
#include <numeric>

std::vector<size_t> sliceVector(const std::vector<size_t>& vector, const size_t first_idx, const size_t second_idx)
{
    if (first_idx >= second_idx || vector.size() < second_idx)
        throw std::invalid_argument("Index mismatch!");

    return std::vector<size_t>(vector.begin() + first_idx, vector.begin() + second_idx);
}

Matrix::Matrix(const size_t nrow, const size_t ncol, const std::vector<double>& data) //constructor, called when an object is created, don't include default vars here
    : m_nrow{ nrow }, m_ncol{ ncol }, m_data{ data }
{
    if ((nrow * ncol) != data.size())
        throw std::invalid_argument("Data size does not match the dimensions!");
    if ((nrow == 0 && ncol > 0) || (nrow > 0 && ncol == 0))
        throw std::invalid_argument("One dimension is zero while the other one is not!");
}

void Matrix::print(std::ostream& stream) const
{
    stream << "Size:" << '\t' << m_data.size() << '\n';
    stream << "Cap:" << '\t' << m_data.capacity() << '\n';
    stream << "nRow:" << '\t' << m_nrow << '\n';
    stream << "nCol:" << '\t' << m_ncol << '\n';
    size_t idx{ 0 };
    for (size_t i{ 0 }; i < std::min(5, static_cast<int>(m_nrow)); i++)
    {
        for (size_t j{ 0 }; j < std::min(5, static_cast<int>(m_ncol)); j++)
        {
            stream << this->at(i, j) << ' ';
            idx++;
        }
        stream << '\n';
    }
    stream << "Top 5 rows printed." << '\n';
    stream << '\n';
}

void Matrix::printDims(std::ostream& stream) const
{
    stream << "Size:" << '\t' << m_data.size() << '\n';
    stream << "Cap:" << '\t' << m_data.capacity() << '\n';
    stream << "nRow:" << '\t' << m_nrow << '\n';
    stream << "nCol:" << '\t' << m_ncol << '\n';
    stream << '\n';
}

size_t Matrix::size() const
{
    return m_data.size();
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

double Matrix::sumElements() const
{
    return accumulate(m_data.begin(), m_data.end(), 0.0);
}

double Matrix::dotProduct(const Matrix& other_matrix) const
{
    if (m_ncol != other_matrix.m_ncol || m_nrow != other_matrix.m_nrow)
        throw std::invalid_argument("Matrices have different dimensions!");

    double result{ 0.0 };

    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        result += m_data[i] * other_matrix.m_data[i];
    }

    return result;
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
    }
}

Matrix Matrix::getRow(const size_t row_idx) const
{
    if (row_idx >= m_nrow)
        throw std::invalid_argument("Index exceeds dimensions!");
    else if (m_nrow == 0)
        throw std::invalid_argument("Matrix is empty!");

    size_t idx_start{ row_idx * m_ncol };
    return Matrix{ 1, m_ncol, std::vector<double>(m_data.begin() + idx_start, m_data.begin() + idx_start + m_ncol) };
}

Matrix Matrix::getRows(const std::vector<size_t> row_idx) const
{
    if (std::max_element(row_idx.begin(), row_idx.end())[0] >= m_nrow)
        throw std::invalid_argument("Index exceeds dimensions!");

    std::vector<double> result{};
    
    for (size_t j : row_idx)
    {
        for (size_t i{ 0 }; i < m_ncol; i++)
        {
            result.push_back(m_data[j * m_ncol + i]);
        }
    }

    return Matrix{ row_idx.size(), m_ncol, result };
}

Matrix Matrix::getCol(const size_t col_idx) const
{
    if (col_idx >= m_ncol)
        throw std::invalid_argument("Index exceeds dimensions!");
    else if (m_ncol == 0)
        throw std::invalid_argument("Matrix is empty!");

    std::vector<double> result(m_nrow);
    for (size_t i{ 0 }; i < m_nrow; i++)
    {
        result[i] = m_data[i * m_ncol + col_idx];
    }
    return Matrix{ m_nrow, 1, result };
}

Matrix Matrix::getCols(const std::vector<size_t> col_idx) const
{
    if (std::max_element(col_idx.begin(), col_idx.end())[0] >= m_ncol)
        throw std::invalid_argument("Index exceeds dimensions!");
    
    std::vector<double> result{};
    for (size_t i{ 0 }; i < m_nrow; i++)
    {
        for (size_t j : col_idx)
        {
            result.push_back(this->operator()(i, j));
        }
    }
    return Matrix{ m_nrow, col_idx.size(), result };
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

const double& Matrix::at(const size_t row_idx, const size_t col_idx) const
{
    if (row_idx >= m_nrow || col_idx >= m_ncol)
        throw std::invalid_argument("Index exceeds dimensions!");
    return m_data.at(row_idx * m_ncol + col_idx);
}

const double& Matrix::at(const size_t idx) const
{
    return m_data.at(idx);
}

bool Matrix::isSquare() const
{
    return (m_nrow == m_ncol);
}

Matrix Matrix::transpose() const
{
    Matrix result{ m_ncol, m_nrow, std::vector<double>(m_data.size()) };
    for (int i{ 0 }; i < m_nrow; i++)
    {
        for (int j{ 0 }; j < m_ncol; j++)
        {
            result.m_data[j * m_nrow + i] = m_data[i * m_ncol + j];
        }
    }
    return result;
}

void Matrix::transposeMe()
{
    *this = this->transpose();
}

void Matrix::zeroMe()
{
    m_data = std::vector<double>(m_data.size(), 0.0);
}

Matrix Matrix::hadamardProduct(const Matrix& other_matrix) const
{
    if (m_ncol != other_matrix.m_ncol || m_nrow != other_matrix.m_nrow)
        throw std::invalid_argument("Matrices have different dimensions!");

    Matrix result{ m_nrow, m_ncol, std::vector<double>(m_data.size()) };

    for (size_t i{ 0 }; i < result.size(); i++)
    {
        result[i] = m_data[i]*other_matrix.m_data[i];
    }

    return result;
}

Matrix Matrix::hadamardProductColumnwise(const Matrix& other_matrix) const
{
    if (m_nrow != other_matrix.m_nrow || other_matrix.m_ncol != 1)
        throw std::invalid_argument("Matrices have different dimensions!");

    Matrix result{ m_nrow, m_ncol, std::vector<double>(m_data.size()) };

    for (size_t i{ 0 }; i < m_nrow; i++)
    {
        for (size_t j{ 0 }; j < m_ncol; j++)
        {
            result.m_data[i * m_ncol + j] = m_data[i * m_ncol + j] * other_matrix.m_data[i];
        }
    }

    return result;
}

Matrix Matrix::zeroButOne(const size_t idx) const
{
    Matrix result{ m_nrow, m_ncol, std::vector<double>(m_data.size(), 0.0) };
    result[idx] = m_data.at(idx);
    return result;
}

Matrix Matrix::zeroButOne(const size_t row_idx, const size_t col_idx) const
{
    if (row_idx >= m_nrow || col_idx >= m_ncol)
        throw std::invalid_argument("Index exceeds dimensions!");

    Matrix result{ m_nrow, m_ncol, std::vector<double>(m_data.size(), 0.0) };
    result[row_idx * m_ncol + col_idx] =  m_data.at(row_idx * m_ncol + col_idx);
    return result;
}

Matrix Matrix::zeroButOneRow(const size_t row_idx) const
{
    if (row_idx >= m_nrow)
        throw std::invalid_argument("Index exceeds dimensions!");

    std::vector<double> result(m_data.size(), 0.0);
    size_t idx_start{ row_idx * m_ncol };

    std::vector<double> src{ std::vector<double>(m_data.begin() + idx_start, m_data.begin() + idx_start + m_ncol) };
    std::copy(src.begin(), src.end(), result.begin() + idx_start);

    return Matrix{ m_nrow, m_ncol, result };
}

std::vector<double> Matrix::columnwiseMean() const
{
    if (m_ncol == 0)
        throw std::invalid_argument("Zero columns!");

    std::vector<double> result{};
    for (size_t i{ 0 }; i < m_ncol; i++)
    {
        std::vector<double> cur_col{ this->getCol(i).m_data };
        double average{ std::accumulate(cur_col.begin(), cur_col.end(), 0.0) / static_cast<double>(cur_col.size()) };
        result.push_back(average);
    }
    return result;
}

std::vector<double> Matrix::columnwiseStdDev() const
{
    if (m_ncol == 0)
        throw std::invalid_argument("Zero columns!");

    std::vector<double> result{};
    for (size_t i{ 0 }; i < m_ncol; i++)
    {
        std::vector<double> cur_col{ this->getCol(i).m_data };
        double average{ std::accumulate(cur_col.begin(), cur_col.end(), 0.0) / static_cast<double>(cur_col.size()) };
        double sum{ 0.0 };
        for (double& element : cur_col)
        {
            element -= average;
            sum += element * element;
        }
        result.push_back(sqrt(sum / cur_col.size()));
    }
    return result;
}

void Matrix::operator+=(const Matrix& other_matrix)
{
    if (m_ncol != other_matrix.m_ncol || m_nrow != other_matrix.m_nrow)
        throw std::invalid_argument("Matrices have different dimensions!");

    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        m_data[i] += other_matrix.m_data[i];
    }
}

void Matrix::operator-=(const Matrix& other_matrix)
{
    if (m_ncol != other_matrix.m_ncol || m_nrow != other_matrix.m_nrow)
        throw std::invalid_argument("Matrices have different dimensions!");

    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        m_data[i] -= other_matrix.m_data[i];
    }
}

void Matrix::operator*=(const double multiplier)
{
    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        m_data[i] *= multiplier;
    }
}

void Matrix::operator+=(const double add_me)
{
    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        m_data[i] += add_me;
    }
}

void Matrix::operator-=(const double subtract_me)
{
    for (size_t i{ 0 }; i < m_data.size(); i++)
    {
        m_data[i] -= subtract_me;
    }
}

Matrix Matrix::operator+(const double add_me) const
{
    Matrix result{ *this };
    result += add_me;
    return result;
}

Matrix Matrix::operator+(const Matrix& other_matrix) const
{
    Matrix result{ *this };
    result += other_matrix;
    return result;
}

Matrix Matrix::operator-(const double subtract_me) const
{
    Matrix result{ *this };
    result -= subtract_me;
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

    Matrix result{ m_nrow, other_matrix.m_ncol, std::vector<double>(m_nrow * other_matrix.m_ncol) };

    for (size_t i{ 0 }; i < m_nrow; i++)
    {
        for (size_t j{ 0 }; j < other_matrix.m_ncol; j++)
        {
            double element_value{ 0.0 };

            for (size_t k{ 0 }; k < m_ncol; k++)
            {
                element_value += m_data[i * m_ncol + k] * other_matrix.m_data[k * other_matrix.m_ncol + j];
            }

            result.m_data[i * other_matrix.m_ncol + j] = element_value;
        }
    }

    return result;
}

Matrix Matrix::addColumnwise(const Matrix& other_matrix) const
{
    if (m_nrow != other_matrix.m_nrow || other_matrix.m_ncol != 1)
        throw std::invalid_argument("Matrices have different dimensions!");

    Matrix result{ *this };

    for (size_t i{ 0 }; i < m_nrow; i++)
    {
        for (size_t j{ 0 }; j < m_ncol; j++)
        {
            result(i, j) += other_matrix[i];
        }
    }

    return result;
}

double& Matrix::operator[](const size_t idx)
{
    return m_data[idx];
}

const double& Matrix::operator[](const size_t idx) const
{
    return m_data[idx];
}

double& Matrix::operator()(const size_t row_idx, const size_t col_idx)
{
    return m_data[row_idx * m_ncol + col_idx];
}

const double& Matrix::operator()(const size_t row_idx, const size_t col_idx) const
{
    return m_data[row_idx * m_ncol + col_idx];
}