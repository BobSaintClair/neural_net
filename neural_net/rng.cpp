#include "rng.h"
#include <stdexcept>

RNG::RNG()
{
    std::random_device m_RandomDevice{};
    m_seed = m_RandomDevice();
    m_randomEngine = std::default_random_engine{ m_seed };
}

double RNG::generateFromNormal(const double mean, const double st_dev)
{
    std::normal_distribution<double> myGaussianDistribution{ mean, st_dev };

    double result{ myGaussianDistribution(m_randomEngine) };

    return result;
}

double RNG::generateFromUniform(const double a, const double b)
{
    std::uniform_real_distribution<double> myUniformDistribution{ a, b };

    double result{ myUniformDistribution(m_randomEngine) };

    return result;
}

int RNG::generateFromUniform(const int a, const int b)
{
    std::uniform_int_distribution<int> myUniformDistribution{ a, b };

    int result{ myUniformDistribution(m_randomEngine) };

    return result;
}

std::vector<int> RNG::generateNDistinctFromUniform(const int a, const int b, const int n)
{
    if (n <= 0)
        throw std::invalid_argument("n is not positive!");
    else if (n > b - a)
        throw std::invalid_argument("Not enough distinct ints in range!");

    std::vector<int> result{};
    result.reserve(b - a + 1);

    for (int i{ a }; i <= b; i++)
    {
        result.push_back(i);
    }

    std::shuffle(result.begin(), result.end(), m_randomEngine);

    result.resize(n);

    return result;
}