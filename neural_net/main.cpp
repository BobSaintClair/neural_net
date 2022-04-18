#include "main.h"

int main()
{	
	data_frame_unlabeled myData{ read_csv2("data.csv") };
	Matrix& y{ myData.first };
	Matrix& x{ myData.second };

	NeuralNet nn{ std::vector<size_t>{ x.nCol(), 30, 10, 5, y.nCol() }};
	nn.train(y, x);
	
	return 0;	
}