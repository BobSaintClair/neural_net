#include "timer.h"
#include <iostream>

void Timer::reset()
{
	m_beg = clock_type::now();
}

double Timer::elapsed() const
{
	return std::chrono::duration_cast<second_type>(clock_type::now() - m_beg).count();
}

void Timer::print_elapsed() const
{
	double elapsed{ std::chrono::duration_cast<second_type>(clock_type::now() - m_beg).count() };
	std::cout << "Time elapsed: " << elapsed << " seconds\n";
}