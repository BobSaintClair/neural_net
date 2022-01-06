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

	Vector identity(const Vector& x)
	{
		return x;
	}
	
	Vector identity_der(const Vector& x)
	{
		return Vector{ x.size(), 1.0 };
	}
	
	Vector sigmoid(const Vector& x)
	{
		Vector f_x{ x.size() };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = sigmoid(x[i]);
		}
		return f_x;
	}

	Vector sigmoid_der(const Vector& x)
	{
		Vector f_x{ x.size() };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = sigmoid_der(x[i]);
		}
		return f_x;
	}

	Vector tanh(const Vector& x)
	{
		Vector f_x{ x.size() };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = tanh(x[i]);
		}
		return f_x;
	}

	Vector tanh_der(const Vector& x)
	{
		Vector f_x{ x.size() };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = tanh_der(x[i]);
		}
		return f_x;
	}

	Vector relu(const Vector& x)
	{
		Vector f_x{ x.size() };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = relu(x[i]);
		}
		return f_x;
	}

	Vector relu_der(const Vector& x)
	{
		Vector f_x{ x.size() };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = relu_der(x[i]);
		}
		return f_x;
	}

	Vector softplus(const Vector& x)
	{
		Vector f_x{ x.size() };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = softplus(x[i]);
		}
		return f_x;
	}

	Vector softplus_der(const Vector& x)
	{
		Vector f_x{ x.size() };
		for (size_t i{ 0 }; i < x.size(); i++)
		{
			f_x[i] = softplus_der(x[i]);
		}
		return f_x;
	}
}