#ifndef KERNEL_H
#define KERNEL_H

#include <iostream>
#include <stdio.h>
#include <cuda.cuh>

using namespace std;

__device__ double evaluate_solution_GPU(int* pos, double* mat, int length, int mat_dim);
__device__ double euclidian_distance_GPU(double x1, double y1, double x2, double y2);
__global__ void calculate_distances(double* new_data, double* distance_matrix, int n);
#endif