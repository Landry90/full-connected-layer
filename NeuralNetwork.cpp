#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(){}
NeuralNetwork::~NeuralNetwork(){}

NeuralNetwork::NeuralNetwork(int num_layers, int num_neurons_per_layer, double alpha){
    this->alpha = alpha;
    this->num_layers = num_layers;
    this->layers.resize(num_layers);
    for(int i=0; i<num_layers; i++){
        this->layers[i] = Layer(num_neurons_per_layer, num_neurons_per_layer);
        this->layers[num_layers-1].setNumNeurons(1);
    }
}

std::vector<double> NeuralNetwork::forward(std::vector<double>& input_datas){
    int data_size = input_datas.size();
    this->layers[0].setPrevLayerSize(data_size);
    layers[0].forward(input_datas);
    for (int i=1; i< this->num_layers; i++){
        std::vector<double> layer_output = layers[i-1].getLayerOutputs();
        layers[i].forward(layer_output);
    }
    
}

std::vector<Layer> NeuralNetwork::getLayers(){
    return this -> layers;
}

void NeuralNetwork::printLayerOutputs(){
    for(size_t i=0; i< this->layers.size(); i++){
        layers[i].printLayerOutputs();
        std::cout<<std::endl;
    }
}

void NeuralNetwork::backward(std::vector<double>&target, double y, double my_alpha){

    //Dans cette section, on calcule l'erreur
    std::vector<double> last_layer_outputs = layers[num_layers-1].getLayerOutputs();
    double p = last_layer_outputs[0];
    double error = lossFunction(p, y);

    // A partir d'ici, on fait la rétro-propagation
    for (int i = this->layers.size()-1; i>0 ; --i){

        int layer_size = layers[i].getLayerSize();
        std::vector<Neuron> neurons = layers[i].getNeurons();
        int prev_layer_size = layers[i].getPrevLayerSize();
        std::vector<double> layer_outputs = layers[i].getLayerOutputs();
        std::vector<double> prev_layer_outputs = layers[i-1].getLayerOutputs();

        // delta est le vecteur gradient d'erreurs des différents neurones de la couche actuelle
        // Son calcule se fait dans la boucle for ci-dessous
        std::vector<double> delta(layer_size);
        for (int j = 0; j < layer_size; ++j){
            delta[j] = activationDerivative(neurons[j].getOutput()) * error;
        }

        std::vector<double> delta_w(prev_layer_size);
        for (int k=0; k < prev_layer_size; ++k){
            for(int m=0; m<layer_size; ++m){
                delta_w[k] = my_alpha * layer_outputs[k] * delta[m];
            }
        }

        
        /*
        for(int j=0; j<neurons.size()-1; i++){
            int prev_layer_size = neurons[j].getWeights().size();
            std::vector<double> vect;

            
        }
        */
    }   
}

/*
std::vector<double>NeuralNetwork::forward(std::vector<double>& input){
    // Vérifier que la taille des entrées correspond au nombre d'entrées de la couche
    if (inputs.size() != this->numInputs) {
        throw std::runtime_error("Le nombre d'entrées ne correspond pas au nombre d'entrées de la couche !");
    }

    // Calculer la sortie de chaque neurone de la couche
    std::vector<double> outputs(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        outputs[i] = neurons[i].computeWeightedSum(inputs);
    }
}
*/