#include "Layer.h"


Layer::Layer(){}
Layer::~Layer(){}
Layer::Layer(int prev_layer_size, int num_neurons){
    this->neurons.resize(num_neurons);
    this->prev_layer_size = prev_layer_size;
    this->num_neurons = num_neurons;
    //this-> neurons.resize(num_neurons);
    for (int i = 0; i < num_neurons; i++) {
        this->neurons[i] = Neuron(prev_layer_size);
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
    this->neurons.resize(num_neurons);
}
int Layer::getNumNeurons(){
    return this-> neurons.size();
}
std::vector<double> Layer::getLayerOutputs(){
    this->layerOutputs.resize(neurons.size());
    for(size_t i=0; i < neurons.size(); i++){
        layerOutputs[i] = this->neurons[i].getOutput();
    }
    return layerOutputs;
}

void Layer::printLayerOutputs(){
    for(size_t i=0 ; i<layerOutputs.size(); i++){
        std::cout<< this->layerOutputs[i] <<std::endl;
    }
}

void Layer::forward(std::vector<double>& prev_layer_values)
{
    for(size_t i=0; i < neurons.size(); i++){
        neurons[i].activate(prev_layer_values);
    }
}

std::vector<Neuron> Layer::getNeurons(){
    return this-> neurons;
}