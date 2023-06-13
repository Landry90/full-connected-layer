#include "Layer.h"


Layer::Layer(){}
Layer::~Layer(){}
Layer::Layer(int prev_layer_size, int num_neurons){
    this->neurons.resize(num_neurons);
    this->prev_layer_size = prev_layer_size;
    this->num_neurons = num_neurons;
    this->delta.resize(this->num_neurons);
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

void Layer::setNumNeurons(int new_num_neurons){
    this->num_neurons = new_num_neurons;
    this->neurons.resize(new_num_neurons);
}
int Layer::getNumNeurons(){
    return this-> neurons.size();
}
std::vector<double> Layer::getLayerOutputs(){
    std::vector<double> layerOutputs(neurons.size());
    for(size_t i=0; i < neurons.size(); i++){
        layerOutputs[i] = this->neurons[i].getOutput();
    }
    return layerOutputs;
}

void Layer::printLayerOutputs(){
    std::vector<double> layerOutputs(neurons.size());
    layerOutputs = this->getLayerOutputs();
    for(size_t i=0 ; i< layerOutputs.size(); i++){
        std::cout<< layerOutputs[i] <<std::endl;
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
/*
std::vector<double> Layer::getDelta(){
    return this-> delta;
}
void Layer::setDelta(std::vector<double>& new_delta){
    this->delta = new_delta;
}
*/

std::vector<std::vector<double>> Layer::getWeightsMatrix(){
    std::vector<std::vector<double>> weights_matrix(this->num_neurons);
    for(int i=0; i < this->num_neurons; i++){
        weights_matrix[i].resize(this->prev_layer_size);
    }
    for(int i=0; i < this->num_neurons; i++){
        std::vector<double> neuronWeigts = this->neurons[i].getWeights();
        for(int j=0; j<neuronWeigts.size(); ++j){
            weights_matrix[i][j] = neuronWeigts[j];
        }
    }
    return weights_matrix;
}