#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int num_layers, int num_neurons_per_layer){
    this->num_layers = num_layers;
    for(int i=0; i<num_layers; i++){
        this->layers[i] = Layer(num_neurons_per_layer, num_neurons_per_layer);
    }
    this->layers[num_layers-1].setNumNeurons(1);
}

NeuralNetwork::NeuralNetwork(){}
NeuralNetwork::~NeuralNetwork(){}

std::vector<double> NeuralNetwork::forward(std::vector<double>& input_datas){
    int data_size = input_datas.size();
    this->layers[0].setPrevLayerSize(data_size);
    layers[0].forward(input_datas);
    for (int i=1; i< this->num_layers; i++){
        std::vector<double> layer_output = layers[i-1].getLayerOutputs();
        layers[i].forward(layer_output);
    }
    
}

/*
std::vector<double>NeuralNetwork::forward(std::vector<double>& input){
    // Vérifier que la taille des entrées correspond au nombre d'entrées de la couche
    if (inputs.size() != this->numInputs) {
        throw std::runtime_error("Le nombre d'entrées ne correspond pas au nombre d'entrées de la couche !");
    }

    // Calculer la sortie de chaque neurone de la couche
    std::vector<double> outputs(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        outputs[i] = neurons[i].computeWeightedSum(inputs);
    }
}
*/