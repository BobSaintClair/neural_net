#include "main.h"

int main()
{		
	data_frame_unlabeled myData{ read_csv2("data2.csv") };
	Matrix& y{ myData.first };
	Matrix& x{ myData.second };
	NeuralNet nn{ std::vector<size_t>{ myData.second.nCol(), 5, myData.first.nCol() }};
	nn.train(y, x);

	return 0;	
}