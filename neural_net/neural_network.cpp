#include "neural_network.h"
#include "rng.h"

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

NeuralNet::NeuralNet(const std::vector<size_t> layers, const double learning_rate, const ActivationFunction hidden_af, const ActivationFunction outer_af)
    : m_layers{ layers }, m_hidden_af{ hidden_af }, m_outer_af{ outer_af }, m_learning_rate{ learning_rate }
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

void NeuralNet::train(Matrix& y, Matrix& x)
{
    if (m_layers.size() == 3)
    {
        RNG rng{};

        int batch_size{ 10000 };
        size_t n_neurons1{ m_layers[1] };
        size_t n_neurons2{ m_layers[2] };

        size_t n_indep_vars{ m_layers[0] };
        size_t n_rows{ x.nRow() };

        x.transposeMe();
        y.transposeMe();

        double orig_error{ 0.0 };
        for (int i{ 0 }; i < x.nCol(); i++)
        {
            Matrix first_layer{ act_hidden(m_weights[0] * x.getCol(i) + m_biases[0]) };
            Matrix second_layer{ act_outer(m_weights[1] * first_layer + m_biases[1]) };
            Matrix delta_y{ (second_layer - y.getCol(i)) };
            orig_error += (delta_y.transpose() * delta_y)[0];
        }
        orig_error *= 1.0 / static_cast<double>(x.nCol());

        for (int i{ 0 }; i < 5000; i++)
        {
            Matrix weights1_grad{ n_neurons1, n_indep_vars, std::vector<double>(n_indep_vars * n_neurons1) };
            Matrix biases1_grad{ n_neurons1, 1, std::vector<double>(n_neurons1) };
            Matrix weights2_grad{ n_neurons2, n_neurons1, std::vector<double>(n_neurons1 * n_neurons2) };
            Matrix biases2_grad{ n_neurons2, 1, std::vector<double>(n_neurons2) };

            std::vector<int> idx = rng.generateNDistinctFromUniform(0, static_cast<int>(n_rows) - 1, batch_size);

            double error{ 0.0 };

            for (int j : idx)
            {
                Matrix x_vec{ x.getCol(j) };
                Matrix y_vec{ y.getCol(j) };

                Matrix first_layer{ act_hidden(m_weights[0] * x_vec + m_biases[0]) };
                Matrix first_layer_der{ act_hidden_der(m_weights[0] * x_vec + m_biases[0]) };
                Matrix second_layer{ act_outer(m_weights[1] * first_layer + m_biases[1]) };
                Matrix second_layer_der{ act_outer_der(m_weights[1] * first_layer + m_biases[1]) };
                Matrix delta_y{ (second_layer - y_vec).transpose() };

                error += (delta_y * delta_y.transpose())[0];

                biases2_grad += delta_y * second_layer_der;

                for (int k = 0; k < weights2_grad.nRow(); k++)
                {
                    for (int l = 0; l < weights2_grad.nCol(); l++)
                    {
                        weights2_grad(k, l) += delta_y[k] * second_layer_der[k] * first_layer[l];
                    }
                }

                for (int k = 0; k < biases1_grad.size(); k++)
                {
                    for (int l = 0; l < delta_y.size(); l++)
                    {
                        biases1_grad[k] += delta_y[l] * second_layer_der[l] * first_layer_der[k] * m_weights[1](l, k);
                    }
                }

                for (int k = 0; k < weights1_grad.nRow(); k++)
                {
                    for (int l = 0; l < weights1_grad.nCol(); l++)
                    {
                        for (int m = 0; m < delta_y.size(); m++)
                        {
                            weights1_grad(k, l) += delta_y[m] * second_layer_der[m] * first_layer_der[k] * m_weights[1](m, k) * x_vec[l];
                        }
                    }
                }
            }

            error *= 1.0 / static_cast<double>(batch_size);
            std::cout << "Epoch:\t" << i << '\t' << "Batch error:\t" << error << '\n';

            weights1_grad *= 2.0 / static_cast<double>(batch_size);
            biases1_grad *= 2.0 / static_cast<double>(batch_size);
            weights2_grad *= 2.0 / static_cast<double>(batch_size);
            biases2_grad *= 2.0 / static_cast<double>(batch_size);

            m_weights[0] -= weights1_grad * m_learning_rate;
            m_biases[0] -= biases1_grad * m_learning_rate;
            m_weights[1] -= weights2_grad * m_learning_rate;
            m_biases[1] -= biases2_grad * m_learning_rate;
        }

        double final_error{ 0.0 };
        for (int i{ 0 }; i < x.nCol(); i++)
        {
            Matrix first_layer{ act_hidden(m_weights[0] * x.getCol(i) + m_biases[0]) };
            Matrix second_layer{ act_outer(m_weights[1] * first_layer + m_biases[1]) };
            Matrix delta_y{ (second_layer - y.getCol(i)) };
            orig_error += (delta_y.transpose() * delta_y)[0];
        }
        final_error *= 1.0 / static_cast<double>(x.nCol());

        std::cout << "Original population error:\t" << orig_error << '\n';
        std::cout << "Final population error:\t\t" << final_error << '\n';
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