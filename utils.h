#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <vector>
#include <time.h>

// activationFunction ---> Fonction sigmoide
// lossFunction ---> Fonction d'entropie croisée binaire

// p est la probabilité calculée par le réseau de neurones
// y est la valeur vraie de la classe
// normalize ---> Normalisation z-score

double activationFunction(double x);
double activationDerivative(double x);

double lossFunction(double p, double y); 
double lossFunctionDerivate(double p, double y);


double moyenne(std::vector<double>& data_vector);
double variance(std::vector<double>& data_vector);
double ecart_type(std::vector<double>& data_vector);

std::vector<double> normalize(std::vector<double>& data_vector);

std::vector<std::vector<double>>vector2Matrix(std::vector<double>& vec, int nrows, int ncols);


#endif //UTILS_H