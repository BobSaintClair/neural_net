#include "main.h"

int main()
{	
	data_frame myData{ read_csv("data.csv") };
	Matrix y = myData.second.getCols(std::vector<size_t>{ 0 });
	Matrix x = myData.second.getCols(std::vector<size_t>{ 1, 2, 3 });
		
	NeuralNet nn{ std::vector<size_t>{ x.nCol(), 30, 10, 5, y.nCol() }};
	nn.train(y, x);
	
	return 0;	
}