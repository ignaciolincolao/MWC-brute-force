#ifndef DATASET_H
#define DATASET_H

#include <kernel.cuh>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
class Dataset {
public:
    float *distMatrix_device = nullptr;
    float * X_device = nullptr;
    float * Y_device = nullptr;
    std::vector<float> X;
    std::vector<float> Y;
    float *distMatrix_host = nullptr;
    Dataset(const std::string &filename, const std::string type="csv", char separator=',');
    ~Dataset();
    void generarDistMatrix();
    void printXY() const;
    void savePointFile() const;
    void copyToHost();
    float getDistanciaHost(int i, int j);
    float getDistanciaDevice(int i, int j);
};


#endif