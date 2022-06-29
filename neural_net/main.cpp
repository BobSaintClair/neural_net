#include "main.h"
#include <numeric>

int main()
{	
	data_frame data_test{ read_csv("data_test.csv") };
	data_frame data_train{ read_csv("data_train.csv") };

	Matrix y_test = data_test.second.getCols(std::vector<size_t>{ 0 });
	Matrix x_test = data_test.second.getCols(std::vector<size_t>{ 1, 2, 3, 4 });

	Matrix y_train = data_train.second.getCols(std::vector<size_t>{ 0 });
	Matrix x_train = data_train.second.getCols(std::vector<size_t>{ 1, 2, 3, 4 });

	std::vector<size_t> network_dimensions{ x_train.nCol(), 20, 10, 5, y_train.nCol() };

	NeuralNet nn{ network_dimensions, ActivationFunction::sigmoid, ActivationFunction::identity };	
	nn.train(y_train, x_train, 0.02, 500, 15);
	
	y_test.print();
		
	double test_error{ 0.0 };
	for (int i = 0; i < y_test.nRow(); i++)
	{
		if (i < 100)
		{
			std::cout << nn.predict(x_test.getRow(i).transpose())[0] << '\n';
			std::cout << y_test.getRow(i)[0] << '\n';
			std::cout << '\n';
			std::cout << '\n';
		}

		test_error += abs(y_test[i] - nn.predict(x_test.getRow(i).transpose())[0]);
	}
	std::cout << "Test error: " << test_error / static_cast<double>(y_test.nRow()) << '\n';
	
	return 0;	
}