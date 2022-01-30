#pragma once
#include "matrix.h"

enum class ActivationFunction
{
	identity,
	sigmoid,
	tanh,
	relu,
	softplus
};

namespace activation_functions
{
	double identity(const double x);
	double identity_der();
	double sigmoid(const double x);
	double sigmoid_der(const double x);
	double tanh(const double x);
	double tanh_der(const double x);
	double relu(const double x);
	double relu_der(const double x);
	double softplus(const double x);
	double softplus_der(const double x);
	Matrix identity(const Matrix& x);
	Matrix identity_der(const Matrix& x);
	Matrix sigmoid(const Matrix& x);
	Matrix sigmoid_der(const Matrix& x);
	Matrix tanh(const Matrix& x);
	Matrix tanh_der(const Matrix& x);
	Matrix relu(const Matrix& x);
	Matrix relu_der(const Matrix& x);
	Matrix softplus(const Matrix& x);
	Matrix softplus_der(const Matrix& x);
}