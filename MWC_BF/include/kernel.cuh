#ifndef KERNEL_H
#define KERNEL_H

#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;
__global__ void computeDistMatrix(float *X, float *Y, float *distMatrix_device, int n);
__device__ float distanciaR2(float x1, float y1, float x2, float y2);
__global__ void evaluate_solution_kernel(int *matrixSolution_device, float *distMatrix_device, float *fitness_device, int nQuorum, int nData, int nSolution);
__global__ void evaluate_solution_kernel_v2(int *matrixSolution_device, float *distMatrix_device, float *fitness_device, int nQuorum, int nData, int nSolution); 
__global__ void evaluate_solution_kernel_v3(int n, int k, float *distMatrix_device, float *fitness_device, int nSolution, int batchIdx);
#endif