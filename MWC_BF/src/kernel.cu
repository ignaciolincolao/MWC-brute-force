#include <kernel.cuh>




__device__ float distanciaR2(float x1, float y1, float x2, float y2) {
    return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

__global__ void computeDistMatrix(float *X, float *Y, float *distMatrix_device, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        distMatrix_device[i * n + j] = distanciaR2(X[i], Y[i], X[j], Y[j]);
    }
}


__global__ void evaluate_solution_kernel(int *matrixSolution_device, float *distMatrix_device, float *fitness_device, int nQuorum, int nData, int nSolution){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure we don't go out of bounds
    if (idx < nSolution) {
        float sum = 0.0f;
        for (int i = 0; i < nQuorum - 1; ++i) {
            for (int j = i + 1; j < nQuorum; ++j) {
                int pos_i = matrixSolution_device[idx * nQuorum + i];
                int pos_j = matrixSolution_device[idx * nQuorum + j];
                sum += distMatrix_device[pos_i * nData + pos_j];
            }
        }
        fitness_device[idx] = sum;
    }
}
