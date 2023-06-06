#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <iostream>
#include <vector>
#include "Layer.h"

class NeuralNetwork
{
public:
  NeuralNetwork(int num_layers, int num_neurons_per_layer, double alpha);
  //NeuralNetwork(std::vector<int> layerSizes);
  NeuralNetwork();
  ~NeuralNetwork();
  //void init(std::vector<int>& layerSizes);
  std::vector<double> forward(std::vector<double>& input_datas);
  void backward(std::vector<double>&target, double y, double p);
  //void train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>>& targets, int numEpochs);
  //std::vector<double> predict(std::vector<double>& inputs, double learning_rate);
  std::vector<Layer> getLayers();
  void printLayerOutputs();

private:
  int num_layers;
  std::vector<Layer> layers;
  double alpha;     //learning rate
  unsigned int epochs;
  double precision;
  
};

#endif // NEURALNETWORK_HPP