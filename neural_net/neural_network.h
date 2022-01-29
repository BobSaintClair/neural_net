#pragma once
#include "math.h"

class NeuralNet
{
private:
    ActivationFunction m_hidden_af{};
    ActivationFunction m_outer_af{};
    std::vector<size_t> m_layers{};
    std::vector<Matrix> m_weights{};
    std::vector<Matrix> m_biases{};
    double m_learning_rate{};

    Matrix act_hidden(const Matrix& x) const;
    Matrix act_outer(const Matrix& x) const;
    Matrix act_hidden_der(const Matrix& x) const;
    Matrix act_outer_der(const Matrix& x) const;

public:
    NeuralNet(const std::vector<size_t> layers, const double learning_rate = 0.01, const ActivationFunction hidden_af = ActivationFunction::relu, const ActivationFunction outer_af = ActivationFunction::identity);
    
    void train(Matrix& y, Matrix& x); //fix it so y and x are const
    Matrix predict(const Matrix& x) const;
};