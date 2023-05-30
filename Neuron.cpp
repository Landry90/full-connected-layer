#include "Neuron.h"

int seed = time(0);
std::default_random_engine gen(seed);
// Création d'une distribution nor  male
double mean = 0.0;                      // Moyenne de la distribution
double stddev = 1.0;                    // Écart-type de la distribution
std::uniform_real_distribution<double> distribution(mean, stddev);


Neuron::Neuron(){}
Neuron::~Neuron(){}

Neuron::Neuron(int prev_layer_size) {
    // Initialisation des poids avec des valeurs aléatoires
    this->weights.resize(prev_layer_size);
    //this->weights = randomVector(prev_layer_size);
    //this->weights = randomVec(prev_layer_size);
    //this->bias = init;
    initRandomBias();
    initRandomWeights(prev_layer_size);
    //this->initRandomBias();
}

double Neuron::weighedBiasedSum(const std::vector<double>& prev_layer_values){
    double weightedSum = 0.0;
    int size = this->weights.size();
    for (unsigned int i=0; i<size; i++){
        weightedSum += weights[i]*prev_layer_values[i];
    }
    return weightedSum+(this->bias);
}
void Neuron::activate(const std::vector<double>& prev_layer_values){
    double sum = weighedBiasedSum(prev_layer_values);
    this->output = activationFunction(sum);
}


double Neuron::getOutput() const{
	return this->output;
}

void Neuron::setWeights(const std::vector<double>&newWeights){
	this->weights = newWeights;
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


