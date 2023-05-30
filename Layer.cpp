#include "Layer.h"


Layer::Layer(){}
Layer::~Layer(){}
Layer::Layer(int prev_layer_size, int num_neurons){
    this-> neurons.resize(num_neurons);
    for (int i = 0; i < num_neurons; i++) {
        neurons[i] = Neuron(prev_layer_size);
    }
}

void Layer::setPrevLayerSize(double prev_layer_size){
    this->prev_layer_size = prev_layer_size;
}

int Layer::getPrevLayerSize(){
    return this->prev_layer_size;
}
int Layer::getLayerSize(){
    return this-> neurons.size();
}

void Layer::setNumNeurons(int num_neurons){
    this->num_neurons = num_neurons;
}
int Layer::getNumNeurons(){
    return this->num_neurons;
}
std::vector<double> Layer::getLayerOutputs(){
    std::vector<double> layerOutputs;
    for(int i=0; i<num_neurons; i++){
        this->layerOutputs[i] = this->neurons[i].getOutput();
    }
    return layerOutputs;
}

void Layer::forward(std::vector<double>& prev_layer_values)
{
    for(int i=0; i < neurons.size(); i++){
        neurons[i].activate(prev_layer_values);
    }
}

std::vector<Neuron> Layer::getNeurons(){
    return this-> neurons;
}