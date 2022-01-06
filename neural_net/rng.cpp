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

    std::vector<int> result(n);

    std::uniform_int_distribution<int> myUniformDistribution{ a, b };

    for (int i{ 0 }; i < n; i++)
    {
        int number_generated{ myUniformDistribution(m_randomEngine) };

        while (std::find(result.begin(), result.end(), number_generated) != result.end())
        {
            number_generated = myUniformDistribution(m_randomEngine);
        }

        result[i] = number_generated;
    }

    return result;
}