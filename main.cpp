#include <iostream>
#include "NeuralNetwork.h"
#include "utils.h"


int main()
{
	int prev_layer_size(4);
	int num_neurons_per_layer(3);
	int num_layers(5);
	std::vector<double>input_datas{1.0, 2.5, 2.1, 3.7, 0.002, 0.0, 7.0, 0.0, 0.3, 3.5};

	std::vector<double>input_data{11111.0, 255000.5, 1488452.1, 488393.7, 2800.002, 555550.0, 787.0, 870.0, 890.3, 30.5};

	std::vector<double> norm_input_datas = normalize(input_data);
	for (double data:norm_input_datas)
	{
		std::cout<<data<<std::endl;
	}

	std::cout<<std::endl;

	NeuralNetwork network = NeuralNetwork(num_layers, num_neurons_per_layer, 0.0001);
	network.forward(norm_input_datas);
	network.printLayerOutputs();

	std::cout<<std::endl;

	network.backward(norm_input_datas, 1.0, 0.0001);


	std::cout<<std::endl;
	std::cout<<"Hello world !"<<std::endl;
	return 0;
}