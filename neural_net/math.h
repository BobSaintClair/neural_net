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
	Vector identity(const Vector& x);
	Vector identity_der(const Vector& x);
	Vector sigmoid(const Vector& x);
	Vector sigmoid_der(const Vector& x);
	Vector tanh(const Vector& x);
	Vector tanh_der(const Vector& x);
	Vector relu(const Vector& x);
	Vector relu_der(const Vector& x);
	Vector softplus(const Vector& x);
	Vector softplus_der(const Vector& x);
}
