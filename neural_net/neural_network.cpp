#include "neural_network.h"
#include "rng.h"
#include <numeric>

Matrix NeuralNet::act_hidden(const Matrix& x) const
{
    switch (m_hidden_af)
    {
    case ActivationFunction::identity:
        return activation_functions::identity(x);
    case ActivationFunction::sigmoid:
        return activation_functions::sigmoid(x);
    case ActivationFunction::tanh:
        return activation_functions::tanh(x);
    case ActivationFunction::relu:
        return activation_functions::relu(x);
    case ActivationFunction::softplus:
        return activation_functions::softplus(x);
    default:
        return activation_functions::relu(x);
    }
}

Matrix NeuralNet::act_outer(const Matrix& x) const
{
    switch (m_outer_af)
    {
    case ActivationFunction::identity:
        return activation_functions::identity(x);
    case ActivationFunction::sigmoid:
        return activation_functions::sigmoid(x);
    case ActivationFunction::tanh:
        return activation_functions::tanh(x);
    case ActivationFunction::relu:
        return activation_functions::relu(x);
    case ActivationFunction::softplus:
        return activation_functions::softplus(x);
    default:
        return activation_functions::identity(x);
    }
}

Matrix NeuralNet::act_hidden_der(const Matrix& x) const
{
    switch (m_hidden_af)
    {
    case ActivationFunction::identity:
        return activation_functions::identity_der(x);
    case ActivationFunction::sigmoid:
        return activation_functions::sigmoid_der(x);
    case ActivationFunction::tanh:
        return activation_functions::tanh_der(x);
    case ActivationFunction::relu:
        return activation_functions::relu_der(x);
    case ActivationFunction::softplus:
        return activation_functions::softplus_der(x);
    default:
        return activation_functions::relu_der(x);
    }
}

Matrix NeuralNet::act_outer_der(const Matrix& x) const
{
    switch (m_outer_af)
    {
    case ActivationFunction::identity:
        return activation_functions::identity_der(x);
    case ActivationFunction::sigmoid:
        return activation_functions::sigmoid_der(x);
    case ActivationFunction::tanh:
        return activation_functions::tanh_der(x);
    case ActivationFunction::relu:
        return activation_functions::relu_der(x);
    case ActivationFunction::softplus:
        return activation_functions::softplus_der(x);
    default:
        return activation_functions::identity_der(x);
    }
}

NeuralNet::NeuralNet(const std::vector<size_t> layers, const double learning_rate, const size_t batch_size, const size_t epochs, const ActivationFunction hidden_af, const ActivationFunction outer_af)
    : m_layers{ layers }, m_hidden_af{ hidden_af }, m_outer_af{ outer_af }, m_learning_rate{ learning_rate }, m_batch_size{ batch_size }, m_epochs{ epochs }, m_n_layers{ layers.size() }
{
    if (layers.size() < 3)
        throw std::invalid_argument("Not enough layers!");

    for (size_t i : layers)
    {
        if (i == 0)
            throw std::invalid_argument("One of the layers is empty!");
    }

    RNG rng{};

    for (size_t i{ 1 }; i < layers.size(); i++)
    {
        Matrix weights{ layers[i], layers[i - 1], std::vector<double>(layers[i] * layers[i - 1]) };
        Matrix biases{ layers[i], 1, std::vector<double>(layers[i]) };

        for (int j{ 0 }; j < layers[i] * layers[i - 1]; j++)
        {
            weights[j] = rng.generateFromNormal(0.0, sqrt(2.0 / static_cast<double>(layers[i - 1])));
        }

        for (int j{ 0 }; j < layers[i]; j++)
        {
            biases[j] = rng.generateFromNormal(0.0, sqrt(2.0 / static_cast<double>(layers[i - 1])));
        }

        m_weights.push_back(weights);
        m_biases.push_back(biases);
    }
}

void NeuralNet::train(const Matrix& y, const Matrix& x)
{
    if (m_layers.at(0) != x.nCol() || m_layers.at(m_n_layers - 1) != y.nCol())
        throw std::invalid_argument("Layer size mismatch!");

    RNG rng{};

    size_t n_rows{ x.nRow() };
    size_t n_iters_per_epoch = static_cast<size_t>(ceil(static_cast<double>(n_rows) / static_cast<double>(m_batch_size)));

    std::vector<size_t> shuffled_batch_idx(n_rows);
    std::iota(shuffled_batch_idx.begin(), shuffled_batch_idx.end(), 0);

    std::vector<Matrix> node_vals(m_n_layers - 1);
    std::vector<Matrix> node_vals_der(m_n_layers - 1);

    std::vector<Matrix> weights_grad{ m_weights };
    std::vector<Matrix> biases_grad{ m_biases };

    for (size_t cur_epoch{ 0 }; cur_epoch < m_epochs; cur_epoch++)
    {
        rng.shuffleVector(shuffled_batch_idx);

        double epoch_error{ 0.0 };

        for (size_t i{ 0 }; i < n_iters_per_epoch; i++)
        {
            for (size_t j{ 0 }; j < weights_grad.size(); j++)
            {
                weights_grad[j].zeroMe();
                biases_grad[j].zeroMe();
            }

            size_t count_me{ 0 };
            for (size_t j{ i * m_batch_size }; j < std::min((i + 1) * m_batch_size, n_rows); j++)
            {
                count_me++;
                Matrix x_vec{ x.getRow(shuffled_batch_idx[j]).transpose() };
                Matrix y_vec{ y.getRow(shuffled_batch_idx[j]).transpose() };

                node_vals[0] = act_hidden(m_weights[0] * x_vec + m_biases[0]);
                node_vals_der[0] = act_hidden_der(m_weights[0] * x_vec + m_biases[0]);

                for (int k{ 1 }; k < node_vals.size() - 1; k++)
                {
                    node_vals[k] = act_hidden(m_weights[k] * node_vals[k - 1] + m_biases[k]);
                    node_vals_der[k] = act_hidden_der(m_weights[k] * node_vals[k - 1] + m_biases[k]);
                }

                node_vals[node_vals.size() - 1] = act_outer(m_weights[node_vals.size() - 1] * node_vals[node_vals.size() - 2] + m_biases[node_vals.size() - 1]);
                node_vals_der[node_vals.size() - 1] = act_outer_der(m_weights[node_vals.size() - 1] * node_vals[node_vals.size() - 2] + m_biases[node_vals.size() - 1]);

                Matrix y_hat = node_vals[node_vals.size() - 1];
                Matrix y_delta = (y_hat - y_vec).transpose();

                epoch_error += (y_delta * y_delta.transpose())[0];

                for (int k{ 0 }; k < weights_grad.size(); k++)
                {
                    for (int l{ 0 }; l < weights_grad[k].nRow(); l++)
                    {
                        Matrix bias_val{ node_vals_der[k].zeroButOne(l) };
                        for (int m{ k + 1 }; m < weights_grad.size(); m++)
                        {
                            bias_val = (m_weights[m] * bias_val).hadamardProduct(node_vals_der[m]);
                        }
                        biases_grad[k][l] += (y_delta * bias_val)[0];

                        if (k == 0)
                        {
                            for (int m{ 0 }; m < weights_grad[k].nCol(); m++)
                            {
                                weights_grad[k](l, m) += (y_delta * (bias_val * x_vec[m]))[0];
                            }
                        }
                        else
                        {
                            for (int m{ 0 }; m < weights_grad[k].nCol(); m++)
                            {
                                weights_grad[k](l, m) += (y_delta * (bias_val * node_vals[k - 1][m]))[0];
                            }
                        }
                    }
                }
            }

            for (int j{ 0 }; j < weights_grad.size(); j++)
            {
                weights_grad[j] *= 2.0 / static_cast<double>(count_me);
                m_weights[j] -= weights_grad[j] * m_learning_rate;

                biases_grad[j] *= 2.0 / static_cast<double>(count_me);
                m_biases[j] -= biases_grad[j] * m_learning_rate;
            }
        }

        std::cout << "Epoch: " << cur_epoch << '\t' << "Error: " << epoch_error / static_cast<double>(n_rows) << '\n';
    }
}

Matrix NeuralNet::predict(const Matrix& x) const
{
    Matrix result{ x };
    for (size_t i{ 0 }; i < m_weights.size() - 1; i++)
    {
        result = act_hidden(m_weights[i] * result + m_biases[i]);
    }
    result = act_outer(m_weights[m_weights.size() - 1] * result + m_biases[m_weights.size() - 1]);
    return result;
}