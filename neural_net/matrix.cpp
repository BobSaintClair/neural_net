#include "matrix.h"

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
    for (size_t i{ 0 }; i < m_nrow; i++)
    {
        for (size_t j{ 0 }; j < m_ncol; j++)
        {
            stream << m_data[idx] << ' ';
            idx++;
        }
        stream << '\n';
    }
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
                element_value += m_data[i * m_ncol + k] * other_matrix.m_data[k * other_matrix.m_ncol + j];
            }

            result.m_data[i * ncol + j] = element_value;
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