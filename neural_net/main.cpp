#include "rng.h"
#include "matrix.h"
#include "math.h"
#include "read_csv.h"
#include "timer.h"
#include <iostream>

double fun_input(Vector weights, Vector x, double bias)
{
	return (weights.dotProduct(x) + bias);
}

int main()
{
	/*
	data_frame_unlabeled myData{ read_csv2("data.csv") };
	Vector& y{ myData.first };
	Matrix& x{ myData.second };

	Vector weights{ std::vector<double>{0.414,0.414,0.414} };

	Vector y_hat{ std::vector<double>(y.size()) };
	for (size_t i{ 0 }; i < y_hat.size(); i++)
	{
		y_hat[i] = 3 * activation_functions::tanh(weights.dotProduct(x.getRow(i)));
	}
	for (int i{ 0 }; i < 30; i++)
	{
		std::cout << y[i] << '\t' << y_hat[i] << '\n';
	}
	*/	
	int batch_size{ 10000 };

	RNG rng{};

	data_frame_unlabeled myData{ read_csv2("data.csv") };
	Vector& y{ myData.first };
	Matrix& x{ myData.second };
	
	//init weights and biases
	size_t indep_vars{ x.nCol() };
	size_t nrows{ x.nRow() };
	Vector weights{ std::vector<double>(indep_vars) };

	for (int i{ 0 }; i < indep_vars; i++)
	{
		weights[i] = rng.generateFromNormal(0.0, 0.8);
	}
	double bias{ rng.generateFromNormal(0.0, 0.8) };




	weights.print();
	for (int k = 0; k < 400; k++)
	{
		Vector weights_grad{ std::vector<double>(indep_vars) };
		double bias_grad{ 0.0 };
		std::vector<int> idx = rng.generateNDistinctFromUniform(0, nrows - 1, batch_size);
		
		for (int i{ 0 }; i < indep_vars; i++)
		{
			for (int j : idx)
			{
				double value_for_af{ x.getRow(j).dotProduct(weights) + bias };
				weights_grad[i] += (3 * activation_functions::tanh(value_for_af) - y.at(j)) * activation_functions::tanh_der(value_for_af) * x.at(j, i);
			}
			weights_grad[i] *= (2.0 / 10000.0);
		}
		
		for (int j : idx)
		{
			double value_for_af{ x.getRow(j).dotProduct(weights) + bias };
			bias_grad += (3 * activation_functions::tanh(value_for_af) - y.at(j)) * activation_functions::tanh_der(value_for_af);
		}
		bias_grad *= (2.0 / 10000.0);

		weights -= weights_grad * 0.02;
		bias -= bias_grad * 0.02;
	}
	weights.print();
	std::cout << bias;
	
	return 0;
}