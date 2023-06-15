#include <kernel.cuh>

__device__ double evaluate_solution_GPU(int* pos, double* mat, int length, int mat_dim) {
    double sum = 0.0;
    for (int i = 0; i <= length - 2; i++) {
        for (int j = i + 1; j <= length - 1; j++) {
            int idx1 = pos[i] * mat_dim + pos[j];
            sum += mat[idx1];
        }
    }
    return sum;
}

__device__ double euclidian_distance_GPU(double x1, double y1, double x2, double y2)
{
	double calculation = pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 1 / (double)2);
	return calculation;
}

__global__ void calculate_distances(double* new_data, double* distance_matrix, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        for (int j = 0; j < n; j++) {
            distance_matrix[i * n + j] = euclidian_distance_GPU(new_data[i * 2 + 0], new_data[i * 2 + 1], new_data[j * 2 + 0], new_data[j * 2 + 1]);
        }
    }
}