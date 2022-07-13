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
    std::pair<std::vector<double>, std::vector<double>> m_norm{};
    size_t m_n_layers{};
    bool m_has_been_trained{};

    Matrix act_hidden(const Matrix& x) const;
    Matrix act_outer(const Matrix& x) const;
    Matrix act_hidden_der(const Matrix& x) const;
    Matrix act_outer_der(const Matrix& x) const;

public:
    NeuralNet(const std::vector<size_t> layers, const ActivationFunction hidden_af, const ActivationFunction outer_af);
    NeuralNet(const std::string filename);

    void train(const Matrix& y_orig, const Matrix& x_orig, const double learning_rate = 0.01, const size_t batch_size = 10000, const size_t epochs = 10, const double l2_reg = 0.0);
    Matrix predict(const Matrix& x_orig) const;
    void save(const std::string filename) const;
    void load(const std::string filename);
    void normalizer(Matrix& x);
    void applyNormalizer(Matrix& x) const;
};