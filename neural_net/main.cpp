#include "rng.h"
#include "matrix.h"
#include "math.h"
#include "read_csv.h"
#include "timer.h"
#include <iostream>

int main()
{
	//data_frame myData{ read_csv("data.csv") };
	Matrix vec1{ 2, 2, std::vector<double>{3.0, 6.0, 3.0, 1.0} };
	Vector vec2{ std::vector<double>{-1.0, 1.0} };
	Vector vec3{ vec1 * vec2 };
	vec1.print();
	vec2.print();
	vec3.print();

	return 0;
}