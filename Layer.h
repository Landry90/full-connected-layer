#ifndef LAYER_H
#define LAYER_H
#include "Neuron.h"
#include <vector>
#include <iostream>

/*
enum LayerType
{
	STANDARD = 0, //Standard layer : fully connected perceptrons
	OUTPUT, // Output : No bias neuron
	INPUT, // Input: Standard input (output of neurons is outputRaw() )
	SOFTMAX //K-Class Classification Layer 

};
*/

//enum ActivationFunction;

class Layer
{
public:
	Layer();
	Layer(int prev_layer_size, int num_neurons);
	~Layer();
	void forward(std::vector<double>& prev_layer_values);
	//void backward(const std::vector<double>& prev_layer_gradients, double learning_rate);
	std::vector<Neuron> getNeurons();
	int getLayerSize();
	void setPrevLayerSize(double prev_layer_size);
	int getPrevLayerSize();
	void setNumNeurons(int new_num_neurons);
	int getNumNeurons();
	std::vector<double> getLayerOutputs();
	void printLayerOutputs();
	//std::vector<double> getDelta();
	//void setDelta(std::vector<double>& new_delta);
	std::vector<std::vector<double>> getWeightsMatrix();

	std::vector<double> delta;
	std::vector<std::vector<double>> weights_matrix;


private:
	int prev_layer_size;
	int num_neurons;
	std::vector<double> layerOutputs;
	std::vector<Neuron> neurons;
};

#endif //LAYER_H