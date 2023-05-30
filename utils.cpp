#include "utils.h"

double activationFunction(double x){
    return 1.0 / (1.0 + std::exp(-x));
}