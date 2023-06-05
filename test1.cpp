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

// Réseau de neurones avec une seule couche cachée
class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, double learningRate) {
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->learningRate = learningRate;

        // Initialisation des poids aléatoirement
        weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                weightsInputHidden[i][j] = (double)rand() / RAND_MAX;
            }
        }

        weightsHiddenOutput.resize(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            weightsHiddenOutput[i] = (double)rand() / RAND_MAX;
        }
    }

    // Propagation avant
    double forwardPropagation(const std::vector<double>& inputs) {
        hiddenLayer.resize(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; ++j) {
                sum += inputs[j] * weightsInputHidden[j][i];
            }
            hiddenLayer[i] = sigmoid(sum);
        }

        double output = 0.0;
        for (int i = 0; i < hiddenSize; ++i) {
            output += hiddenLayer[i] * weightsHiddenOutput[i];
        }
        return sigmoid(output);
    }

    // Rétropropagation
    void backwardPropagation(double y_true, double y_pred, const std::vector<double>& inputs) {
        double lossGradient = (y_pred - y_true) / (y_pred * (1 - y_pred));  // Gradient de la fonction de perte

        std::vector<double> deltaHidden(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            deltaHidden[i] = lossGradient * weightsHiddenOutput[i] * sigmoidDerivative(hiddenLayer[i]);
        }

        // Mise à jour des poids de la couche cachée vers la sortie
        for (int i = 0; i < hiddenSize; ++i) {
            weightsHiddenOutput[i] -= learningRate * lossGradient * hiddenLayer[i];
        }

        // Mise à jour des poids de la couche d'entrée vers la couche cachée
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                weightsInputHidden[i][j] -= learningRate * deltaHidden[j] * inputs[i];
            }
        }
    }

private:
    int inputSize;
    int hiddenSize;
    double learningRate;
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<double> weightsHiddenOutput;
    std::vector<double> hiddenLayer;
};
