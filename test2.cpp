#include <iostream>
#include <cmath>
#include <vector>

// Fonction sigmoïde
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Dérivée de la fonction sigmoïde
double sigmoidDerivative(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

// Fonction de perte - Entropie croisée binaire
double binaryCrossEntropy(double y_true, double y_pred) {
    return -((y_true * log(y_pred)) + ((1 - y_true) * log(1 - y_pred)));
}

// Réseau de neurones avec plusieurs couches cachées
class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, const std::vector<int>& hiddenSizes, double learningRate) {
        this->inputSize = inputSize;
        this->hiddenSizes = hiddenSizes;
        this->learningRate = learningRate;
        this->numLayers = hiddenSizes.size();

        // Initialisation des poids aléatoirement
        weightsInputHidden.resize(numLayers);
        for (int layer = 0; layer < numLayers; ++layer) {
            int inputSizeLayer = (layer == 0) ? inputSize : hiddenSizes[layer - 1];
            int outputSizeLayer = hiddenSizes[layer];
            weightsInputHidden[layer].resize(inputSizeLayer, std::vector<double>(outputSizeLayer));
            for (int i = 0; i < inputSizeLayer; ++i) {
                for (int j = 0; j < outputSizeLayer; ++j) {
                    weightsInputHidden[layer][i][j] = (double)rand() / RAND_MAX;
                }
            }
        }

        weightsHiddenOutput.resize(hiddenSizes.back());
        for (int i = 0; i < hiddenSizes.back(); ++i) {
            weightsHiddenOutput[i] = (double)rand() / RAND_MAX;
        }
    }

    // Propagation avant
    double forwardPropagation(const std::vector<double>& inputs) {
        std::vector<std::vector<double>> hiddenLayers(numLayers);
        hiddenLayers[0].resize(hiddenSizes[0]);
        for (int i = 0; i < hiddenSizes[0]; ++i) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; ++j) {
                sum += inputs[j] * weightsInputHidden[0][j][i];
            }
            hiddenLayers[0][i] = sigmoid(sum);
        }

        for (int layer = 1; layer < numLayers; ++layer) {
            hiddenLayers[layer].resize(hiddenSizes[layer]);
            for (int i = 0; i < hiddenSizes[layer]; ++i) {
                double sum = 0.0;
                for (int j = 0; j < hiddenSizes[layer - 1]; ++j) {
                    sum += hiddenLayers[layer - 1][j] * weightsInputHidden[layer][j][i];
                }
                hiddenLayers[layer][i] = sigmoid(sum);
            }
        }

        double output = 0.0;
        for (int i = 0; i < hiddenSizes.back(); ++i) {
            output += hiddenLayers.back()[i] * weightsHiddenOutput[i];
        }
        return sigmoid(output);
    }

    // Rétropropagation
    void backwardPropagation(double y_true, double y_pred, const std::vector<double>& inputs) {
        double deltaOutput = (y_pred - y_true) * sigmoidDerivative(y_pred);
        std::vector<std::vector<double>> deltaHidden(numLayers);

        deltaHidden.back().resize(hiddenSizes.back());
        for (int i = 0; i < hiddenSizes.back(); ++i) {
            deltaHidden.back()[i] = deltaOutput * weightsHiddenOutput[i] * sigmoidDerivative(hiddenLayer[i]);
        }

        for (int layer = numLayers - 2; layer >= 0; --layer) {
            deltaHidden[layer].resize(hiddenSizes[layer]);
            for (int i = 0; i < hiddenSizes[layer]; ++i) {
                double sum = 0.0;
                for (int j = 0; j < hiddenSizes[layer + 1]; ++j) {
                    sum += deltaHidden[layer + 1][j] * weightsInputHidden[layer + 1][i][j];
                }
                deltaHidden[layer][i] = sum * sigmoidDerivative(hiddenLayer[i]);
            }
        }

        // Mise à jour des poids de la couche cachée vers la sortie
        for (int i = 0; i < hiddenSizes.back(); ++i) {
            weightsHiddenOutput[i] -= learningRate * deltaOutput * hiddenLayer[i];
        }

        // Mise à jour des poids de la couche d'entrée vers la première couche cachée
        for (int layer = 0; layer < numLayers; ++layer) {
            for (int i = 0; i < (layer == 0 ? inputSize : hiddenSizes[layer - 1]); ++i) {
                for (int j = 0; j < hiddenSizes[layer]; ++j) {
                    weightsInputHidden[layer][i][j] -= learningRate * deltaHidden[layer][j] * (layer == 0 ? inputs[i] : hiddenLayer[i]);
                }
            }
        }
    }

private:
    int inputSize;
    std::vector<int> hiddenSizes;
    int numLayers;
    double learningRate;
    std::vector<std::vector<std::vector<double>>> weightsInputHidden;
    std::vector<double> weightsHiddenOutput;
    std::vector<double> hiddenLayer;
};

int main() {
    // Exemple d'utilisation
    int inputSize = 2;
    std::vector<int> hiddenSizes = {3, 4};  // Deux couches cachées avec 3 et 4 neurones respectivement
    double learningRate = 0.1;

    NeuralNetwork neuralNetwork(inputSize, hiddenSizes, learningRate);

    // Données d'entraînement
    std::vector<std::vector<double>> trainingData = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};

    // Entraînement du réseau
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (const auto& data : trainingData) {
            std::vector<double> inputs(data.begin(), data.begin() + inputSize);
            double y_true = data.back();

            double y_pred = neuralNetwork.forwardPropagation(inputs);
            neuralNetwork.backwardPropagation(y_true, y_pred, inputs);
        }
    }

    // Test du réseau entraîné
    std::cout << "Prédictions du réseau :" << std::endl;
    for (const auto& data : trainingData) {
        std::vector<double> inputs(data.begin(), data.begin() + inputSize);
        double y_true = data.back();

        double y_pred = neuralNetwork.forwardPropagation(inputs);
        std::cout << "Entrée : " << inputs[0] << ", " << inputs[1] << " | Sortie prédite : " << y_pred << " | Sortie attendue : " << y_true << std::endl;
    }

    return 0;
}
