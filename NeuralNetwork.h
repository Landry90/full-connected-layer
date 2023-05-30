#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <iostream>
#include <vector>
#include "Layer.h"

class NeuralNetwork
{
public:
  NeuralNetwork(int num_layers, int num_neurons_per_layer);
  //NeuralNetwork(std::vector<int> layerSizes);
  NeuralNetwork();
  ~NeuralNetwork();
  //void init(std::vector<int>& layerSizes);
  std::vector<double> forward(std::vector<double>& input_datas);
  //void backward(std::vector<double>&target)
  //void train(std::vector<std::vector<double>> inputs, std::vector<std::vector<douoble>>& targets, int numEpochs);
  //std::vector<double> predict(std::vector<double>& inputs, double learning_rate);
  std::vector<Layer> getLayers();

private:
  int num_layers;
  std::vector<Layer> layers;
  double learning_rate;
  unsigned int epochs;
  double precision;
  
};

#endif // NEURALNETWORK_HPP