#include "Neuron.h"

int seed = time(0);
std::default_random_engine gen(seed);
// Création d'une distribution nor  male
double mean = -0.01;                      // Moyenne de la distribution
double stddev = 0.01;                    // Écart-type de la distribution
std::uniform_real_distribution<double> distribution(mean, stddev);


Neuron::Neuron(){}
Neuron::~Neuron(){}

Neuron::Neuron(int prev_layer_size) {
    // Initialisation des poids avec des valeurs aléatoires
    this->weights.resize(prev_layer_size);
    initRandomBias();
    initRandomWeights(prev_layer_size);
}

double Neuron::weighedBiasedSum(std::vector<double>& prev_layer_values){
    double weightedSum = 0.0;
    int size = this->weights.size();
    for (int i=0; i<size; i++){
        weightedSum += this->weights[i] * prev_layer_values[i];
    }
    return weightedSum + (this->bias);
}
void Neuron::activate(std::vector<double>& prev_layer_values){
    double sum = weighedBiasedSum(prev_layer_values);
    this->output = activationFunction(sum);
}


double Neuron::getOutput(){
	return this->output;
}

void Neuron::setWeights(std::vector<double>& new_weights){
	this->weights = new_weights;
}


void Neuron::setBias(double newBias){
	this-> bias = newBias;
}

double Neuron::getBias(){
    return this->bias;
}

std::vector<double> Neuron::getWeights(){
    return this-> weights;
}
void Neuron::printWeights(){
    for (auto elm:this->weights)
        std::cout<<elm<<std::endl;
}

void Neuron::initRandomWeights(int prev_layer_size){
    for (int i=0; i<prev_layer_size; i++){
        this->weights[i] = distribution(gen);
    }
}

void Neuron::initRandomBias(){
    this->bias = distribution(gen);
}


