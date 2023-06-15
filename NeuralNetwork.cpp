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

void NeuralNetwork::forward(std::vector<double>& input_datas){
    int data_size = input_datas.size();
    this->layers[0].setPrevLayerSize(data_size);
    layers[0].forward(input_datas);
    for (int i=1; i< this->num_layers; i++){
        std::vector<double> layer_output = layers[i-1].getLayerOutputs();
        this->layers[i].forward(layer_output);
    }
    
}

std::vector<Layer> NeuralNetwork::getLayers(){
    return this -> layers;
}

void NeuralNetwork::printLayerOutputs(){
    for(Layer layer : this->layers){
        layer.printLayerOutputs();
    }
}

void NeuralNetwork::backward(std::vector<double>& input_datas, double y, double my_alpha){

    //Dans cette section, on calcule l'erreur
    Layer last_layer = this->layers[num_layers-1];
    std::vector<double> last_layer_outputs = last_layer.getLayerOutputs();
    double p = last_layer_outputs[0];
    double error = lossFunction(p, y);

    // Dans cette section, on fait la mise à jour des poids de l'unique neurone de la couche de sortie.
    std::vector<Neuron> last_layer_neurons = last_layer.getNeurons();
    std::vector<double> last_layer_weights = last_layer_neurons[0].getWeights();

    double delta_last = error*activationDerivative(p);
    std::vector<double> last_prev_layer_outputs = layers[this->num_layers-2].getLayerOutputs();
    for(int a=0; a<last_layer_weights.size(); ++a){
        last_layer_weights[a] += my_alpha*delta_last*last_prev_layer_outputs[a];
    }
    this->layers[num_layers-1].delta[0] = delta_last;
    
   // Dans cette section, on calcule le gradient d'erreurs pour chaque couche cachée.
    for(int i=this->layers.size()-2; i>=0; --i){
        std::vector<double> layer_outputs = layers[i].getLayerOutputs();
        std::vector<std::vector<double>>next_layer_weights_matrix;
        next_layer_weights_matrix = this->layers[i+1].getWeightsMatrix();

        double som = 0.0;
        std::vector<double> derivated_output(layers[i].getLayerSize());
        for(int j=0; j< this->layers[i].getLayerSize(); ++j){
            for(int k=0; k < this->layers[i+1].getLayerSize(); ++k){
                som += next_layer_weights_matrix[k][j] * (layers[i+1].delta)[k];
            }
            derivated_output[j] = activationDerivative(layer_outputs[j]);
            (this->layers[i].delta)[j] = som * derivated_output[j];
        }

    }

    // Mise à jour des poids et biais des couches cachées à partir de la deuxième

    for(int i = this->layers.size()-2; i>0; --i){
        std::vector<Neuron> layer_neurons(layers[i].getLayerSize());
        layer_neurons = this->layers[i].getNeurons();
        std::vector<double> delta_layer(layers[i].getLayerSize());
        delta_layer = this->layers[i].delta;
        std::vector<std::vector<double>> layer_neurons_weights(this->layers[i-1].getLayerSize());
        std::vector<double> prev_layer_outputs(this->layers[i-1].getLayerSize());

        // On extrait les poids et les ouptputs de chaque neurone de la couche
        for(int j=0; j < this->layers[i-1].getLayerSize(); ++j){
            layer_neurons_weights[j] = layer_neurons[j].getWeights();  // On extrait le vecteur de poids du neuron j
            prev_layer_outputs[j] = layer_neurons[j].getOutput();  // On extrait le ouotput du neuron j
        }

        // Mise à jour des poids

        // On calcule le vecteur correspondant à la somme entre delta, les valeurs de la couche précéente et alpha
        std::vector<double> dw(this->layers[i].getLayerSize(), 0.0);
        for(int k=0; k < this->layers[i].getLayerSize(); ++k){
            for(int m=0; m < this->layers[i-1].getLayerSize(); ++m){
                dw[k] = my_alpha * delta_layer[k] * prev_layer_outputs[m];
            }    
        }
         // On calcule la nouvelle matrice de poids de la couche
        for(int n=0; n < this->layers[i-1].getLayerSize(); ++n){
            for(int p=0; p < layers[i].getLayerSize(); ++p){
                layer_neurons_weights[n][p] += dw[p];
            }
        }

        // On met à jour la matrice de poids de la couche
        for(int t=0; t < layers[i].getLayerSize(); ++t){
            layer_neurons[t].setWeights(layer_neurons_weights[t]);
        }

        // Mise à jour des biais

        // On calcule les nouveaux biais des neurones de la couche courante
        std::vector<double> layer_bias(this->layers[i].getLayerSize());
        for(int r=0; r < this->layers[i].getLayerSize(); ++r){
            layer_bias[r] = layer_neurons[r].getBias();
        }

        // On met à jour les biais des neurones de la couche
        for(int r=0; r < layers[i].getLayerSize(); ++r){
            layer_bias[r] += my_alpha * delta_layer[r];
            layer_neurons[r].setBias(layer_bias[r]);
        } 
    }

    // Mise à jour des poids et des biais de la première couche cachée
    // Pour ce faire, on répète le même processus que le précédent aux différences que:
    // il n'y a pas de boucle pour parcourir plusieurs couches
    // On ne considère que la couche d'indice 0
    // La couche précédente est remplacée par le vecteur de données

    // Mise à jour des poids
    std::vector<Neuron> layer_neurons(layers[0].getLayerSize());
    layer_neurons = this->layers[0].getNeurons();
    std::vector<double> delta_layer(this->layers[0].getLayerSize());
    delta_layer = this->layers[0].delta;
    std::vector<std::vector<double>> layer_neurons_weights(this->layers[0].getLayerSize());
    std::vector<double> prev_layer_outputs(input_datas.size());

    // On extrait les poids de la premiere couche cachee
    for(int j=0; j < this->layers[0].getLayerSize(); ++j){
        layer_neurons_weights[j].resize(input_datas.size());  // On extrait le vecteur de poids du neuron j
        for(int k=0; k < input_datas.size(); ++k){
            layer_neurons_weights[j][k] = (layer_neurons[j].getWeights())[k];
        }
    }
    prev_layer_outputs = input_datas;

    // Mise à jour des poids

    // On calcule le vecteur correspondant au produit entre delta, les valeurs de la couche précédente et alpha
    std::vector<double> dw(input_datas.size(), 0.0);
    for(int k=0; k < this->layers[0].getLayerSize(); ++k){
        for(int m=0; m < input_datas.size(); ++m){
            dw[k] = my_alpha * delta_layer[k] * prev_layer_outputs[m];
        }
    }

    // On calcule la nouvelle matrice de poids de la couche
    for(int n=0; n < layers[0].getLayerSize(); ++n){
        for(int p=0; p < input_datas.size(); ++p){
            layer_neurons_weights[n][p] += dw[p];
        }
    }

    // On met à jour la matrice de poids de la couche
    for(int t=0; t < layers[0].getLayerSize(); ++t){
        layer_neurons[t].setWeights(layer_neurons_weights[t]);
    }

    // Mise à jour des biais
    // On calcule les nouveaux biais des neurones de la couche
    std::vector<double> layer_bias(this->layers[0].getLayerSize());
    for(int r=0; r < this->layers[0].getLayerSize(); ++r){
        layer_bias[r] = layer_neurons[r].getBias();
    }

    // On met à jour les biais des neurones de la couche
    for(int r=0; r < this->layers[0].getLayerSize(); ++r){
        layer_bias[r] += my_alpha * delta_layer[r];
        layer_neurons[r].setBias(layer_bias[r]);
    }
}

/*
void NeuralNetwork::train(std::vector<double>& x_train, std::vector<double>& y_train, int n_epochs, double alpha){

    this->forward(x_train);
}
*/