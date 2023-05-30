#include <iostream>
#include "NeuralNetwork.h"


int main()
{
	int num_neurons(10);
	int num_layers(3);
	std::vector<double> tab = {0.3, 0.2, 0.1, 0.5899, 0.99993};
	NeuralNetwork(num_layers, num_neurons);
	std::cout<<"Hello world !!"<<std::endl;


	/*
	Layer my_layer = Layer(tab.size(), num_neurons);
	my_layer.forward(tab);
	std::vector<Neuron> neurons = my_layer.getNeurons();
	for (int i=0; i<num_neurons; i++){
		std::cout<<neurons[i].getOutput()<<std::endl;
	}
	std::cout<<"Hello World !!"<<std::endl;
	*/

	/*
	Neuron my_neuron = Neuron(tab.size());
	std::cout<<"Bias = " << my_neuron.getBias()<<std::endl;
	std::cout<<"Hello World !!"<<std::endl<<std::endl<<std::endl;
	
	my_neuron.activate(tab);
	std::cout<<"Output = "<<my_neuron.getOutput()<<std::endl;
	std::cout<<"Hello World !!"<<std::endl;
	*/
	return 0;
}