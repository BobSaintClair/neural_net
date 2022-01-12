#include "main.h"

int main()
{
	data_frame_unlabeled myData{ read_csv2("data2.csv") };
	Matrix& y{ myData.first };
	Matrix& x{ myData.second };

	RNG rng{};

	int batch_size{ 10000 };
	double learning_rate{ 0.01 };
	size_t n_neurons1{ 4 };
	size_t n_neurons2{ y.nCol() };

	size_t n_indep_vars{ x.nCol() };
	size_t n_rows{ x.nRow() };

	x.transposeMe();
	y.transposeMe();
	
	Matrix weights1{ n_neurons1, n_indep_vars, std::vector<double>(n_indep_vars * n_neurons1) };
	for (int i{ 0 }; i < n_indep_vars * n_neurons1; i++)
	{
		weights1[i] = rng.generateFromNormal(0.0, sqrt(2.0 / static_cast<double>(n_indep_vars)));
	}

	Matrix biases1{ n_neurons1, 1, std::vector<double>(n_neurons1) };
	for (int i{ 0 }; i < n_neurons1; i++)
	{
		biases1[i] = rng.generateFromNormal(0.0, sqrt(2.0 / static_cast<double>(n_indep_vars)));
	}

	Matrix weights2{ n_neurons2, n_neurons1, std::vector<double>(n_neurons1 * n_neurons2) };
	for (int i{ 0 }; i < n_neurons1 * n_neurons2; i++)
	{
		weights2[i] = rng.generateFromNormal(0.0, sqrt(2.0 / static_cast<double>(n_neurons1)));
	}

	Matrix biases2{ n_neurons2, 1, std::vector<double>(n_neurons2) };
	for (int i{ 0 }; i < n_neurons2; i++)
	{
		biases2[i] = rng.generateFromNormal(0.0, sqrt(2.0 / static_cast<double>(n_neurons1)));
	}
	
	double orig_error{ 0.0 };
	for (int i{ 0 }; i < x.nCol(); i++)
	{
		Matrix first_layer{ activation_functions::relu(weights1 * x.getCol(i) + biases1) };
		Matrix second_layer{ activation_functions::identity(weights2 * first_layer + biases2) };
		Matrix delta_y{ (second_layer - y.getCol(i)) };
		orig_error += (delta_y.transpose() * delta_y)[0];
	}
	orig_error *= 1.0 / static_cast<double>(x.nCol());

	for (int i{ 0 }; i < 2000; i++)
	{
		Matrix weights1_grad{ n_neurons1, n_indep_vars, std::vector<double>(n_indep_vars * n_neurons1) };
		Matrix biases1_grad{ n_neurons1, 1, std::vector<double>(n_neurons1) };
		Matrix weights2_grad{ n_neurons2, n_neurons1, std::vector<double>(n_neurons1 * n_neurons2) };
		Matrix biases2_grad{ n_neurons2, 1, std::vector<double>(n_neurons2) };

		std::vector<int> idx = rng.generateNDistinctFromUniform(0, static_cast<int>(n_rows) - 1, batch_size);

		double error{ 0.0 };

		for (int j : idx)
		{
			Matrix first_layer{ activation_functions::relu(weights1 * x.getCol(j) + biases1) };
			Matrix first_layer_der{ activation_functions::relu_der(weights1 * x.getCol(j) + biases1) };
			Matrix second_layer{ activation_functions::identity(weights2 * first_layer + biases2) };
			Matrix second_layer_der{ activation_functions::identity_der(weights2 * first_layer + biases2) };
			Matrix delta_y{ (second_layer - y.getCol(j)).transpose() };

			error += (delta_y * delta_y.transpose())[0];

			biases2_grad += delta_y * second_layer_der;

			for (int k = 0; k < weights2_grad.nRow(); k++)
			{
				for (int l = 0; l < weights2_grad.nCol(); l++)
				{
					weights2_grad.at(k, l) += delta_y[k] * second_layer_der[k] * first_layer[l];
				}
			}

			for (int k = 0; k < biases1_grad.size(); k++)
			{
				for (int l = 0; l < delta_y.size(); l++)
				{
					biases1_grad[k] += delta_y[l] * second_layer_der[l] * first_layer_der[k] * weights2.at(l, k);
				}
			}

			for (int k = 0; k < weights1_grad.nRow(); k++)
			{
				for (int l = 0; l < weights1_grad.nCol(); l++)
				{
					for (int m = 0; m < delta_y.size(); m++)
					{
						weights1_grad.at(k, l) += delta_y[m] * second_layer_der[m] * first_layer_der[k] * weights2.at(m, k) * x.getCol(j)[l];
					}
				}
			}
		}

		error *= 1.0 / static_cast<double>(batch_size);
		std::cout << "Epoch:\t" << i << '\t' << "Batch error:\t" << error << '\n';

		weights1_grad *= 2.0 / static_cast<double>(batch_size);
		biases1_grad *= 2.0 / static_cast<double>(batch_size);
		weights2_grad *= 2.0 / static_cast<double>(batch_size);
		biases2_grad *= 2.0 / static_cast<double>(batch_size);

		weights1 -= weights1_grad * learning_rate;
		biases1 -= biases1_grad * learning_rate;
		weights2 -= weights2_grad * learning_rate;
		biases2 -= biases2_grad * learning_rate;
	}

	double final_error{ 0.0 };
	for (int i{ 0 }; i < x.nCol(); i++)
	{
		Matrix first_layer{ activation_functions::relu(weights1 * x.getCol(i) + biases1) };
		Matrix second_layer{ activation_functions::identity(weights2 * first_layer + biases2) };
		Matrix delta_y{ (second_layer - y.getCol(i)) };
		final_error += (delta_y.transpose() * delta_y)[0];
	}
	final_error *= 1.0 / static_cast<double>(x.nCol());

	std::cout << "Original population error:\t" << orig_error << '\n';
	std::cout << "Final population error:\t\t" << final_error << '\n';

	for (int j{ 0 }; j < 20; j++)
	{
		Matrix first_layer{ activation_functions::relu(weights1 * x.getCol(j) + biases1) };
		Matrix second_layer{ activation_functions::identity(weights2 * first_layer + biases2) };
		std::cout << y[j] << '\t' << second_layer[0] << '\n';
	}
	
	return 0;
}