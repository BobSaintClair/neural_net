#include "timer.h"

void Timer::reset()
{
	m_beg = clock_type::now();
}

double Timer::elapsed() const
{
	return std::chrono::duration_cast<second_type>(clock_type::now() - m_beg).count();
}

void Timer::print_elapsed(std::ostream& stream) const
{
	double elapsed{ std::chrono::duration_cast<second_type>(clock_type::now() - m_beg).count() };
	stream << "Time elapsed: " << elapsed << " seconds\n";
}