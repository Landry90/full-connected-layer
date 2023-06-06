#include <iostream>
#include "NeuralNetwork.h"


int main()
{
	int prev_layer_size(4);
	int num_neurons_per_layer(3);
	int num_layers(5);
	std::vector<double>input_datas{1.0, 2.5, 2.1, 3.7, 0.002, 0.0, 7.0, 0.0, 0.3, 3.5};

	NeuralNetwork network = NeuralNetwork(num_layers, num_neurons_per_layer, 0.0001);
	
	network.forward(input_datas);
	network.printLayerOutputs();

	std::cout<<"Hello world !"<<std::endl;
	return 0;
}