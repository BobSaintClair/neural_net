#include "rng.h"
#include <random>

double generateFromNormal(const double mean, const double st_dev)
{
    static std::random_device myRandomDevice{};
    static unsigned int seed{ myRandomDevice() };
    static std::default_random_engine myRandomEngine{ seed };

    std::normal_distribution<double> myGaussianDistribution{ mean, st_dev };

    double result{ myGaussianDistribution(myRandomEngine) };

    return result;
}