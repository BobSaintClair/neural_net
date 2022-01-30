#include "math.h"
#include <cmath>

namespace activation_functions
{
	double identity(const double x)
	{
		return x;
	}

	double identity_der()
	{
		return 1.0;
	}

	double sigmoid(const double x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	double sigmoid_der(const double x)
	{
		double f_x{ sigmoid(x) };
		return (f_x * (1 - f_x));
	}

	double tanh(const double x)
	{
		return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
	}

	double tanh_der(const double x)
	{
		double f_x{ tanh(x) };
		return (1 - f_x * f_x);
	}

	double relu(const double x)
	{
		if (x <= 0)
			return 0.0;
		else
			return x;
	}

	double relu_der(const double x)
	{
		if (x <= 0)
			return 0.0;
		else
			return 1.0;
	}

	double softplus(const double x)
	{
		return std::log(1 + exp(x));
	}

	double softplus_der(const double x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	Matrix identity(const Matrix& x)
	{
		return x;
	}

	Matrix identity_der(const Matrix& x)
	{
		return Matrix{ x.nRow(), x.nCol(), std::vector<double>(x.size(), 1.0) };
	}

	Matrix sigmoid(const Matrix& x)
	{
		Matrix f_x{ x.nRow(), x.nCol(), std::vector<double>(x.size()) };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = sigmoid(x[i]);
		}
		return f_x;
	}

	Matrix sigmoid_der(const Matrix& x)
	{
		Matrix f_x{ x.nRow(), x.nCol(), std::vector<double>(x.size()) };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = sigmoid_der(x[i]);
		}
		return f_x;
	}

	Matrix tanh(const Matrix& x)
	{
		Matrix f_x{ x.nRow(), x.nCol(), std::vector<double>(x.size()) };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = tanh(x[i]);
		}
		return f_x;
	}

	Matrix tanh_der(const Matrix& x)
	{
		Matrix f_x{ x.nRow(), x.nCol(), std::vector<double>(x.size()) };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = tanh_der(x[i]);
		}
		return f_x;
	}

	Matrix relu(const Matrix& x)
	{
		Matrix f_x{ x.nRow(), x.nCol(), std::vector<double>(x.size()) };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = relu(x[i]);
		}
		return f_x;
	}

	Matrix relu_der(const Matrix& x)
	{
		Matrix f_x{ x.nRow(), x.nCol(), std::vector<double>(x.size()) };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = relu_der(x[i]);
		}
		return f_x;
	}

	Matrix softplus(const Matrix& x)
	{
		Matrix f_x{ x.nRow(), x.nCol(), std::vector<double>(x.size()) };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = softplus(x[i]);
		}
		return f_x;
	}

	Matrix softplus_der(const Matrix& x)
	{
		Matrix f_x{ x.nRow(), x.nCol(), std::vector<double>(x.size()) };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = softplus_der(x[i]);
		}
		return f_x;
	}
}