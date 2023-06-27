#include <Coalition.cuh>


// Función para comprobar errores de CUDA
inline void checkCudaErrors(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "Error en: " << file << ":" << line << " código: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERRORS(err) checkCudaErrors(err, __FILE__, __LINE__)

Coalition::Coalition(int nQuorum, int nData,float *distMatrix_device, int nBlock, int nThread):
    distMatrix_device(distMatrix_device), nQuorum(nQuorum),nData(nData),nBlock(nBlock),nThread(nThread){
    nSolution = nBlock*nThread;
    X.resize(nQuorum, 0.0f);
    Y.resize(nQuorum, 0.0f);
    solution.resize(nQuorum, 0);
    bestSolution.resize(nQuorum);
    fitness = DBL_MAX;
    bestFitness = DBL_MAX;
    X_device = nullptr;
    Y_device = nullptr;
    fitness_device = nullptr;
    matrixSolution_device = nullptr;
    matrixSolution_host = (int*) malloc(nQuorum * nSolution * sizeof(int));
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&X_device, X.size() * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&Y_device, Y.size() * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&fitness_device, nSolution * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&matrixSolution_device, nQuorum * nSolution * sizeof(int)));
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}

Coalition::~Coalition(){
    CHECK_CUDA_ERRORS(cudaFree(X_device));
    CHECK_CUDA_ERRORS(cudaFree(Y_device));
    CHECK_CUDA_ERRORS(cudaFree(fitness_device));
    CHECK_CUDA_ERRORS(cudaFree(matrixSolution_device));
    free(matrixSolution_host);
}

void Coalition::BestSolution(){
    vector<int> combination;
    string bitmask(nQuorum, 1); // K leading 1's
    bitmask.resize(nData, 0); // N-K trailing 0's
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device); // get the current device id
    cudaGetDeviceProperties(&prop, device);
    int count = 0;
    long long int count_calc = 0;
    do {

        for (int i = 0; i < nData; ++i) { // [0..N-1] integers
            if (bitmask[i]) combination.push_back(i);
        }
        std::copy(combination.begin(), combination.end(), matrixSolution_host + (count % nSolution) * nQuorum);
        combination.clear();
        combination.shrink_to_fit();
        count++;
    
        if (count % nSolution == 0) {
            //auto initial_time = chrono::high_resolution_clock::now();
            CHECK_CUDA_ERRORS(cudaMemcpy(matrixSolution_device, matrixSolution_host, nQuorum * nSolution * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERRORS(cudaGetLastError());
            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
            int threadsPerBlock = nThread;
            int blocksPerGrid = nBlock;
            //cout << "block: " << blocksPerGrid << " - " << count << endl;
            evaluate_solution_kernel<<<blocksPerGrid, threadsPerBlock>>>(matrixSolution_device, distMatrix_device, fitness_device, nQuorum, nData, nSolution);
            CHECK_CUDA_ERRORS(cudaGetLastError());
            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
            find_min_index(nSolution, count, count_calc);
            CHECK_CUDA_ERRORS(cudaGetLastError());
            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
            count=0;
            count_calc++;
            //cout << count_calc << endl;
            //auto final_time = chrono::high_resolution_clock::now();
            //double time_taken = chrono::duration_cast<chrono::nanoseconds>(final_time - initial_time).count();
            //time_taken *= 1e-9;
            //cout << "Time:"<< fixed << time_taken << setprecision(9) << count_calc << endl;
            //cout << "Coalition:" << endl;
        }
    } while (prev_permutation(bitmask.begin(), bitmask.end()));

    CHECK_CUDA_ERRORS(cudaMemcpy(matrixSolution_device, matrixSolution_host, nQuorum * nSolution * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    int threadsPerBlock = nThread;
    int blocksPerGrid = nBlock;
    //cout << "block: " << blocksPerGrid << " - " << count << endl;
    evaluate_solution_kernel<<<blocksPerGrid, threadsPerBlock>>>(matrixSolution_device, distMatrix_device, fitness_device, nQuorum, nData, nSolution);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    find_min_index(nSolution, count,count_calc);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	cout << "Combinaciones:" << count<<endl;
}


void Coalition::find_min_index(int n, int count,int count_calc) {
    thrust::device_ptr<float> fitness_device_ptr(fitness_device);

    thrust::device_ptr<float> min_ptr = thrust::min_element(fitness_device_ptr, fitness_device_ptr + n);

    int min_index = thrust::distance(fitness_device_ptr, min_ptr);
    float min_value = *min_ptr;
    if(min_value < bestFitness){
        bestFitness = min_value;
        CHECK_CUDA_ERRORS(cudaMemcpy(&bestSolution[0], matrixSolution_device + min_index * nQuorum, nQuorum * sizeof(int), cudaMemcpyDeviceToHost));
        cout << count  << " || " <<  count_calc << " || " << bestFitness << endl;
    }
}


