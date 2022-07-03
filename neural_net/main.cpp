#include "main.h"

int main()
{	
	data_frame data_test{ read_csv("data_test.csv") };
	data_frame data_train{ read_csv("data_train.csv") };

	Matrix y_test = data_test.second.getCols(std::vector<size_t>{ 0 });
	Matrix x_test = data_test.second.getCols(std::vector<size_t>{ 1, 2, 3, 4 });

	Matrix y_train = data_train.second.getCols(std::vector<size_t>{ 0 });
	Matrix x_train = data_train.second.getCols(std::vector<size_t>{ 1, 2, 3, 4 });

	std::vector<size_t> network_dimensions{ x_train.nCol(), 20, 10, 5, y_train.nCol() };

	NeuralNet nn{ network_dimensions, ActivationFunction::tanh, ActivationFunction::identity };	
	nn.train(y_train, x_train, 0.02, 1000, 40);
	
	Matrix yhat_test{ nn.predict(x_test) };

	double test_error{ ((yhat_test - y_test).dotProduct(yhat_test - y_test))/y_test.size() };
	
	std::cout << "Test error: " << test_error << '\n';
	
	return 0;	
}