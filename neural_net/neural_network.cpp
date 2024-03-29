#include "neural_network.h"
#include "rng.h"
#include <numeric>
#include <fstream>
#include <sstream>

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

NeuralNet::NeuralNet(const std::vector<size_t> layers, const ActivationFunction hidden_af, const ActivationFunction outer_af)
    : m_layers{ layers }, m_hidden_af{ hidden_af }, m_outer_af{ outer_af }, m_n_layers{ layers.size() }
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

NeuralNet::NeuralNet(const std::string filename)
{
    this->load(filename);
}

void NeuralNet::train(const Matrix& y_orig, const Matrix& x_orig, const double learning_rate, const size_t batch_size, const size_t epochs, const double l2_reg)
{
    Matrix y{ y_orig };
    Matrix x{ x_orig };

    //Ensure the data parameters match the network specifications
    if (m_layers.at(0) != x.nCol() || m_layers.at(m_n_layers - 1) != y.nCol())
        throw std::invalid_argument("Layer size mismatch!");

    //Normalize inputs
    normalizer(x);

    //Initialize random number generator
    RNG rng{};

    //Initialize some useful variables
    size_t n_rows{ x.nRow() };
    size_t n_iters_per_epoch = static_cast<size_t>(ceil(static_cast<double>(n_rows) / static_cast<double>(batch_size)));

    //Create an index vector which will be used to loop through the data
    std::vector<size_t> shuffled_batch_idx(n_rows);
    //Populate the index vector from 0 to n_rows
    std::iota(shuffled_batch_idx.begin(), shuffled_batch_idx.end(), 0);

    //Initialize the node value vector of matrices
    std::vector<Matrix> node_vals(m_n_layers - 1);
    std::vector<Matrix> node_vals_der(m_n_layers - 1);

    //Initialize the gradient and bias gradient matrices
    std::vector<Matrix> weights_grad{ m_weights };
    std::vector<Matrix> biases_grad{ m_biases };

    //Loop through epochs
    for (size_t cur_epoch{ 0 }; cur_epoch < epochs; cur_epoch++)
    {
        //Shuffle the index vector
        rng.shuffleVector(shuffled_batch_idx);
        //Epoch error
        double epoch_error{ 0.0 };

        //Loop through iterations within the epoch
        for (size_t i{ 0 }; i < n_iters_per_epoch; i++)
        {
            //Zero the gradients
            for (size_t j{ 0 }; j < weights_grad.size(); j++)
            {
                weights_grad[j].zeroMe();
                biases_grad[j].zeroMe();
            }

            //Get the indexes used in this iteration
            std::vector<size_t> iter_idx{ sliceVector(shuffled_batch_idx, i * batch_size, std::min((i + 1) * batch_size, n_rows)) };
            
            Matrix x_vec{ x.getRows(iter_idx).transpose() };
            Matrix y_vec{ y.getRows(iter_idx).transpose() };

            //Calculates node values at each layer
            node_vals[0] = act_hidden((m_weights[0] * x_vec).addColumnwise(m_biases[0]));
            node_vals_der[0] = act_hidden_der((m_weights[0] * x_vec).addColumnwise(m_biases[0]));
            for (int k{ 1 }; k < node_vals.size() - 1; k++)
            {
                node_vals[k] = act_hidden((m_weights[k] * node_vals[k - 1]).addColumnwise(m_biases[k]));
                node_vals_der[k] = act_hidden_der((m_weights[k] * node_vals[k - 1]).addColumnwise(m_biases[k]));
            }
            node_vals[node_vals.size() - 1] = act_outer((m_weights[node_vals.size() - 1] * node_vals[node_vals.size() - 2]).addColumnwise(m_biases[node_vals.size() - 1]));
            node_vals_der[node_vals.size() - 1] = act_outer_der((m_weights[node_vals.size() - 1] * node_vals[node_vals.size() - 2]).addColumnwise(m_biases[node_vals.size() - 1]));

            //Get the current prediction
            Matrix y_hat{ node_vals[node_vals.size() - 1] };
            //Difference between predicted and actuals
            Matrix y_delta{ y_hat - y_vec };

            //Add to the epoch error
            epoch_error += y_delta.dotProduct(y_delta);

            //Loop through the layers
            for (int k{ 0 }; k < weights_grad.size(); k++)
            {
                //Loop through rows of the weight matrix 
                for (int l{ 0 }; l < weights_grad[k].nRow(); l++)
                {                 
                    //Calculate the derivative of y_hat w.r.t. bias
                    Matrix bias_der{ node_vals_der[k].zeroButOneRow(l) };
                    for (int m{ k + 1 }; m < weights_grad.size(); m++)
                    {
                        bias_der = (node_vals_der[m]).hadamardProduct(m_weights[m] * bias_der);
                    }
                    biases_grad[k][l] = y_delta.dotProduct(bias_der);
                                        
                    if (k == 0)
                    {
                        for (int m{ 0 }; m < weights_grad[k].nCol(); m++)
                        {
                            weights_grad[k](l, m) = y_delta.dotProduct(bias_der.hadamardProduct(x_vec.getRow(m)));
                        }
                    }
                    else
                    {
                        for (int m{ 0 }; m < weights_grad[k].nCol(); m++)
                        {
                            weights_grad[k](l, m) = y_delta.dotProduct(bias_der.hadamardProduct(node_vals[k - 1].getRow(m)));
                        }
                    }
                }
            }            
            //Calculate the number of data points used in this iteration
            double denominator{ static_cast<double>(iter_idx.size()) };

            //Update the weights and biases
            for (int j{ 0 }; j < weights_grad.size(); j++)
            {
                weights_grad[j] *= 2.0 / denominator;
                //If regularization is positive we regularize
                if (l2_reg > 0.0)
                    m_weights[j] -= (weights_grad[j] * learning_rate + m_weights[j] * (l2_reg * learning_rate));
                else
                    m_weights[j] -= (weights_grad[j] * learning_rate);

                biases_grad[j] *= 2.0 / denominator;
                m_biases[j] -= biases_grad[j] * learning_rate;                
            }
        }
        //Output the epoch number and error
        std::cout << "Epoch: " << cur_epoch + 1 << '\t' << "Error: " << epoch_error / static_cast<double>(n_rows * y.nCol()) << '\n';
    }
}

Matrix NeuralNet::predict(const Matrix& x_orig) const
{
    Matrix x{ x_orig };
    applyNormalizer(x);
    Matrix result{ x.transpose() };
    for (size_t i{ 0 }; i < m_weights.size() - 1; i++)
    {
        result = act_hidden((m_weights[i] * result).addColumnwise(m_biases[i]));
    }
    result = act_outer((m_weights[m_weights.size() - 1] * result).addColumnwise(m_biases[m_weights.size() - 1]));
    return result.transpose();
}

void NeuralNet::save(const std::string filename) const
{
    //Create an output stream class to operate on files
    std::ofstream outfile(filename);

    //Save network activation functions
    outfile << static_cast<size_t>(m_hidden_af) << ',';
    outfile << static_cast<size_t>(m_outer_af) << ',';
    outfile << '\n';

    //Save network dimensions
    for (size_t i{ 0 }; i < m_layers.size(); i++)
    {
        outfile << m_layers[i] << ',';
    }

    outfile << '\n';

    //Save network weights
    for (size_t i{ 0 }; i < m_weights.size(); i++)
    {
        for (size_t j{ 0 }; j < m_weights[i].size(); j++)
        {
            outfile << m_weights[i][j] << ',';
        }
    }

    outfile << '\n';

    //Save network biases
    for (size_t i{ 0 }; i < m_biases.size(); i++)
    {
        for (size_t j{ 0 }; j < m_biases[i].size(); j++)
        {
            outfile << m_biases[i][j] << ',';
        }
    }
}

void NeuralNet::load(const std::string filename)
{
    m_layers.clear();
    m_weights.clear();
    m_biases.clear();
    m_n_layers = 0;

    //Create an input stream class to operate on files
    std::ifstream infile(filename);

    if (!infile.good())
        throw std::invalid_argument("Couldn't open file!");

    std::string line{ "" };
    double data{ 0.0 };
    size_t data_size_t{ 0 };

    std::getline(infile, line);
    std::istringstream iss_act{ line };
    iss_act >> data_size_t;
    m_hidden_af = static_cast<ActivationFunction>(data_size_t);
    iss_act.ignore();
    iss_act >> data_size_t;
    m_outer_af = static_cast<ActivationFunction>(data_size_t);
    iss_act.str(std::string());
    iss_act.clear();

    std::getline(infile, line);
    std::istringstream iss_layers{ line };
    while (iss_layers >> data_size_t)
    {
        if (iss_layers.peek() == ',')
            iss_layers.ignore();

        m_layers.push_back(data_size_t);
    }
    iss_layers.str(std::string());
    iss_layers.clear();

    m_n_layers = m_layers.size();

    for (size_t i{ 1 }; i < m_n_layers; i++)
    {
        Matrix weights{ m_layers[i], m_layers[i - 1], std::vector<double>(m_layers[i] * m_layers[i - 1]) };
        Matrix biases{ m_layers[i], 1, std::vector<double>(m_layers[i]) };

        m_weights.push_back(weights);
        m_biases.push_back(biases);
    }
        
    std::getline(infile, line);
    std::istringstream iss_weights{ line };
    for (size_t i{ 0 }; i < m_weights.size(); i++)
    {
        for (size_t j{ 0 }; j < m_weights[i].size(); j++)
        {
            if (iss_weights.peek() == ',')
                iss_weights.ignore();

            iss_weights >> data;
            m_weights[i][j] = data;
        }
    }
    iss_weights.str(std::string());
    iss_weights.clear();

    std::getline(infile, line);
    std::istringstream iss_biases{ line };
    for (size_t i{ 0 }; i < m_biases.size(); i++)
    {
        for (size_t j{ 0 }; j < m_biases[i].size(); j++)
        {
            if (iss_biases.peek() == ',')
                iss_biases.ignore();

            iss_biases >> data;
            m_biases[i][j] = data;
        }
    }
    iss_biases.str(std::string());
    iss_biases.clear();    
}

void NeuralNet::normalizer(Matrix& x)
{
    m_norm.first = x.columnwiseMean();
    m_norm.second = x.columnwiseStdDev();

    for (double std_dev : m_norm.second)
    {
        if (std_dev == 0.0)
        {
            throw std::invalid_argument("Standard devaition is zero for one of the variables!");
        }
    }

    for (size_t row{ 0 }; row < x.nRow(); row++)
    {
        for (size_t col{ 0 }; col < x.nCol(); col++)
        {
            x(row, col) = (x(row, col) - m_norm.first[col]) / m_norm.second[col];
        }
    }
}

void NeuralNet::applyNormalizer(Matrix& x) const
{
    for (size_t row{ 0 }; row < x.nRow(); row++)
    {
        for (size_t col{ 0 }; col < x.nCol(); col++)
        {
            x(row, col) = (x(row, col) - m_norm.first[col]) / m_norm.second[col];
        }
    }
}