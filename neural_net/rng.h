#pragma once
#include <random>

class RNG
{
private:
    unsigned int m_seed{};
    std::default_random_engine m_randomEngine{};

public:
    RNG();

    double generateFromNormal(const double mean, const double st_dev);
    double generateFromUniform(const double a, const double b);
    int generateFromUniform(const int a, const int b); //includes a and b
    std::vector<int> generateNDistinctFromUniform(const int a, const int b, const int n); //includes a and b
};