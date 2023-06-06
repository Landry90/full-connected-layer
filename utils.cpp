#include "utils.h"


double activationFunction(double x){
    return 1.0 / (1.0 + std::exp(-x));
}

double activationDerivative(double x) {
    return activationFunction(x) * (1 - activationFunction(x));

}
double lossFunction(double p, double y){
    return -(y * log(p) + (1 - y) * log(1 - p));
}

double lossFunctionDerivate(double p, double y){
    return (p - y) / (p * (1 - p));
}

double moyenne(std::vector<double> data_vector){
    double som = 0;
    for(int i=0; i<data_vector.size(); ++i){
        som += data_vector[i];
    }
    return som / data_vector.size();
}

double variance(std::vector<double>& data_vector){
    double moy = moyenne(data_vector);
    double val=0;
    for(int i=0; i<data_vector.size(); ++i){
        val += (data_vector[i] - moy) * (data_vector[i] - moy);
    }
    return val / data_vector.size();
}

double ecart_type(std::vector<double>& data_vector){
    double var = variance(data_vector);
    return sqrt(var);
}
std::vector<double> normalize(std::vector<double>& data_vector){
    for(int i=0; i<data_vector.size(); ++i){
        data_vector[i] = (data_vector[i] - moyenne(data_vector)) / ecart_type(data_vector);
    }
    return data_vector;
}