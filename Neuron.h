#ifndef NEURON_H
#define NEURON_H

#include "utils.h"
#include <vector>
#include <random>
#include <time.h>
#include <iostream>

/*
enum ActivationFunction
{
	LINEAR,
	SIGMOID,
	RELU
};
*/

class Neuron
{
public:
	Neuron();
	Neuron(int prev_layer_size);
	~Neuron();
	double getOutput();
	void initRandomWeights(int prev_layer_size);
	void initRandomBias();
	void setWeights(std::vector<double>&newWeights);
	void setBias(double newBias);
	double getBias();
	std::vector<double> getWeights();
	void printWeights();
	double weighedBiasedSum(std::vector<double>& prev_layer_values);
	void activate(std::vector<double>& prev_layer_values);

	//void setValue(unsigned int layer, unsigned int neuron, double new_value);
	//double getValue(unsigned int layer, unsigned int neuron);
	//void initWeigths();
	//void Forward();
private:
	double output;
	double bias;
	std::vector<double> weights;
	//ActivationFunction activation_function;

/*
	Layer* layer = NULL;
	std::vector<Neuron> left_neurons;
	std::vector<Neuron> right_neurons;
	Layer* layer = NULL;
*/
	
};

#endif //NEURON_H