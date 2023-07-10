#ifndef COALITION_H
#define COALITION_H

#include <iostream>
#include <vector>
#include <kernel.cuh>
#include <boost/dynamic_bitset.hpp>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <chrono>
#include <iomanip>

using namespace std;

class Coalition {
public:
    // Local Variables
    std::vector<float> X;
    std::vector<float> Y;
    std::vector<int> solution;
    std::vector<int> bestSolution;
    float bestFitness;
    float fitness;
    int nData;
    int nQuorum;
    int nBlock;
    int nThread;
    int nSolution;
    int *matrixSolution_host;
    // Cuda Variables
    float *X_device;
    float *Y_device;
    float *fitness_device;
    int *matrixSolution_device;
    float *distMatrix_device;
    Coalition(int nQuorum, int nData,float *distMatrix_device, int nBlock, int nThread);
    ~Coalition();
    void BestSolution();
    void BestSolution2();
    void find_min_index(int n, int count, int count_calc);
    void find_min_index2(int n, int count, int count_calc);
};

#endif