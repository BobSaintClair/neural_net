#include "main.h"

int main()
{	
	std::string train_filename{ "data_train_1.csv" };
	std::string test_filename{ "data_test_1.csv" };

	Matrix y_test{};
	Matrix x_test{};
	Matrix y_train{};
	Matrix x_train{};

	if (std::filesystem::exists(train_filename))
	{
		data_frame data_train{ read_csv(train_filename) };

		y_train = data_train.second.getCols(std::vector<size_t>{ 0 });
		x_train = data_train.second.getCols(std::vector<size_t>{ 1, 2, 3, 4 });
	}
	if (std::filesystem::exists(test_filename))
	{
		data_frame data_test{ read_csv(test_filename) };

		y_test = data_test.second.getCols(std::vector<size_t>{ 0 });
		x_test = data_test.second.getCols(std::vector<size_t>{ 1, 2, 3, 4 });
	}
	
	std::vector<size_t> network_dimensions{ x_train.nCol(), 20, 10, 5, y_train.nCol() };
	NeuralNet nn{ network_dimensions, ActivationFunction::tanh, ActivationFunction::identity };	
	nn.train(y_train, x_train, 0.01, 1000, 200, 0.0005);
	
	Matrix yhat_test{ nn.predict(x_test) };
	double test_error{ ((yhat_test - y_test).dotProduct(yhat_test - y_test))/static_cast<double>(y_test.nRow()) };	
	std::cout << "Test error: " << test_error << '\n';
	
	return 0;	
}