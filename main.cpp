#include <iostream>
#include "NeuralNetwork.h"
#include "utils.h"

int seeds = time(0);
std::default_random_engine gens(seeds);
// Création d'une distribution nor  male
double means = 0.0;                      // Moyenne de la distribution
double stds = 255.0;                    // Ecart-type de la distribution
std::uniform_real_distribution<double> distributions(means, stds);

std::vector<double> initRandomVec (int vec_size){
	std::vector<double> vec(vec_size);
	for (int i=0; i<vec_size; i++){
		vec[i] = distributions(gens);
 	}
    return vec;
}

int main()
{
	int prev_layer_size(4);
	int num_neurons_per_layer(10);
	int num_layers(12);
	int nrows(5);
	int ncols(5);
	int vec_size(100);
	std::vector<double> vec(vec_size);
	vec = initRandomVec(vec_size);

	std::vector<double> vect = read_txt("test.txt");
	std::cout<<"vect.size() = "<<vect.size()<<std::endl;

	std::vector<double>input_datas{1.0, 2.5, 2.1, 3.7, 0.002, 0.0, 7.0, 0.0, 0.3};

	std::vector<double>input_data{11111.0, 255000.5, 1488452.1, 488393.7, 2800.002, 555550.0, 787.0, 870.0, 890.3, 30.5, 12, 5.3, 7.002, 12.009, 3.1, 3.0, 3.0, 0.0, 0.0, 175974.3, 123.3, 987.2, 789.445, 7893.44785, 123.79};

	std::vector<double> norm_input_datas = normalize(vect);

	NeuralNetwork network = NeuralNetwork(num_layers, num_neurons_per_layer, 0.0001);
	network.forward(norm_input_datas);

	network.backward(norm_input_datas, 1.0, 0.0001);
/*
	std::vector<std::vector<double>> mat(nrows);
	for(int i=0; i<nrows; ++i){
		mat[i].resize(ncols);
	}
	mat = vector2Matrix(vec, nrows, ncols);

*/
	std::cout<<"Hello world !"<<std::endl;

	return 0;
}