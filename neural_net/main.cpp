#include "rng.h"
#include "matrix.h"
#include "math.h"
#include "read_csv.h"
#include "timer.h"
#include <iostream>
/*
double fun_input(Vector weights, Vector x, double bias)
{
	return (weights.dotProduct(x) + bias);
}
*/
int main()
{
	data_frame myData{ read_csv("data.csv") };
	print_data_frame(myData);
	/*
	Vector weights{ std::vector<double>(5) };
	Vector weights_grad{ std::vector<double>(5) };

	for (int i{ 0 }; i < weights.size(); i++)
	{
		weights[i] = generateFromNormal(0.0, 1.0);
	}

	double bias{ generateFromNormal(0.0, 1.0) };

	for (int i{ 0 }; i < weights.size(); i++)
	{
		double gradient{  };
	}
	*/
	return 0;
}