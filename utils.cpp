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

double moyenne(std::vector<double>& data_vector){
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

std::vector<std::vector<double>>vector2Matrix(std::vector<double>& vec, int nrows, int ncols){
    std::vector<std::vector<double>> mat(nrows);
    for(int i=0; i<nrows; ++i){
        mat[i].resize(ncols);
    }
    int p = 0; 
    for(int i=0; i<nrows; ++i){
        for(int j=0; j<ncols; ++j){
            mat[i][j] = vec[p];
            ++p;
        }
    }
    return mat;
}

std::vector<double>read_txt(std::string file_path){
    std::vector<double> temp;
    std::ifstream f(file_path);
    if (f.is_open()){
        int dim;
        f >> dim;
        std::cout<< dim << std::endl;
        temp.resize(dim);
        int k = 0;
        while (k<dim){
            f >> temp[k];
            k++;
        }
    }
    else{
        exit(EXIT_FAILURE);
    }
    return temp;

}

/*
 static vector<int> readVector() {
            vector<int> temp;
            ifstream f("test.txt");
            if (f.is_open())
            {
                int dim;
                f >> dim;
                cout << dim << endl;
                temp.resize(dim);
                int k = 0;
                while (k<dim)
                {
                    f >> temp[k];
                    k++;
                }
            }
            else
            {
                exit(EXIT_FAILURE);
            }
            return temp;
        }
*/