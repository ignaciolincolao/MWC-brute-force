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

__global__ void evaluate_solution_kernel_v2(int *matrixSolution_device, float *distMatrix_device, float *fitness_device, int nQuorum, int nData, int nSolution){
    extern __shared__ float shared_distMatrix[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x;

    // Copy distMatrix_device to shared memory
    if (local_idx < nData * nData) {
        shared_distMatrix[local_idx] = distMatrix_device[local_idx];
    }
    __syncthreads();
    // Ensure we don't go out of bounds
    if (idx < nSolution) {
        float sum = 0.0f;
        for (int i = 0; i < nQuorum - 1; ++i) {
            for (int j = i + 1; j < nQuorum; ++j) {
                int pos_i = matrixSolution_device[idx * nQuorum + i];
                int pos_j = matrixSolution_device[idx * nQuorum + j];
                sum += shared_distMatrix[pos_i * nData + pos_j];
            }
        }
        fitness_device[idx] = sum;
    }
}



__global__ void evaluate_solution_kernel_v3(int n, int k, float *distMatrix_device, float *fitness_device, int nSolution, int batchIdx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x + batchIdx * blockDim.x * gridDim.x;
    int f = idx;
    if (idx >= nSolution) {
        return;  // Ensure we don't go out of bounds
    }

    // Generate the idx-th combination using the Combinadic algorithm
    int a = n;
    int b = k;
    int x = (1 << k) - 1;

    while (idx--) {
        b = k;
        while (x & (1 << (a - 1))) {
            x -= (1 << (a - 1));
            --a;
            --b;
        }
        x |= ((1 << (a - 1)) - 1);
        x &= ~((1 << (a - b - 1)) - 1);
    }

    // Calculate the sum of the pairwise distances for the points in the combination
    float sum = 0.0f;
    for (int i = 0; i < n - 1; ++i) {
        if (!(x & (1 << i))) continue; // Skip if point i is not in the combination
        for (int j = i + 1; j < n; ++j) {
            if (x & (1 << j)) {  // If points i and j are in the combination
                sum += distMatrix_device[i * n + j];
            }
        }
    }

    fitness_device[f] = sum;
}
