#include <Dataset.cuh>

// Función para comprobar errores de CUDA
inline void checkCudaErrors(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "Error en: " << file << ":" << line << " código: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERRORS(err) checkCudaErrors(err, __FILE__, __LINE__)


Dataset::Dataset(const std::string &filename, const std::string type, char separator) {
    if(type == "csv" || type == "CSV"){
        std::ifstream inFile(filename);
        if (!inFile) {
            std::cerr << "No se pudo abrir el archivo " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        float x, y;
        char actualSeparator;
        while (inFile >> x >> actualSeparator >> y) {
            if (actualSeparator != separator) {
                std::cerr << "El delimitador no coincide con " << separator << std::endl;
                exit(EXIT_FAILURE);
            }
            X.push_back(x);
            Y.push_back(y);
        }
    }
    else{
        if(type == "json" || type == "JSON"){
            ifstream file(filename);
            if (!file) {
                std::cerr << "No se pudo abrir el archivo " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            json data = json::parse(file);
            cout << "entro" << endl;
            for (size_t i=0; i <data["rollcalls"][0]["votes"].size(); i++){
                X.push_back(data["rollcalls"][0]["votes"][i]["x"]);
                Y.push_back(data["rollcalls"][0]["votes"][i]["y"]);
            }
        }
        else{
            cout << "Tipo de archivo no soportado" << endl;
            exit(EXIT_FAILURE);
        }
    }
    
    int n = X.size();


    CHECK_CUDA_ERRORS(cudaMalloc((void**)&X_device, X.size() * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&Y_device, Y.size() * sizeof(float)));

    CHECK_CUDA_ERRORS(cudaMemcpy(X_device, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(Y_device, Y.data(), Y.size() * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERRORS(cudaMalloc((void**)&distMatrix_device, n * n * sizeof(float)));
    distMatrix_host = (float*) malloc(n * n * sizeof(float));
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    generarDistMatrix();
}

Dataset::~Dataset() {
    CHECK_CUDA_ERRORS(cudaFree(distMatrix_device));
    CHECK_CUDA_ERRORS(cudaFree(X_device));
    CHECK_CUDA_ERRORS(cudaFree(Y_device));
    free(distMatrix_host);
}


void Dataset::generarDistMatrix() {
    int n = X.size();
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    computeDistMatrix<<<numBlocks, threadsPerBlock>>>(X_device, Y_device, distMatrix_device, n);
    
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    copyToHost();
}

void Dataset::printXY() const {
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << "X: " << X[i] << ", Y: " << Y[i] << std::endl;
    }
}


void Dataset::copyToHost() {
    int n = X.size();
    CHECK_CUDA_ERRORS(cudaMemcpy(distMatrix_host, distMatrix_device, n * n * sizeof(float), cudaMemcpyDeviceToHost));
}

float Dataset::getDistanciaHost(int i, int j) {
    int n = X.size();
    if (i < 0 || i >= n || j < 0 || j >= n) {
        std::cerr << "Índice fuera de rango." << std::endl;
        return -1.0f;
    }
    return distMatrix_host[i * n + j];
}

float Dataset::getDistanciaDevice(int i, int j){
    int n = X.size();
    if (i < 0 || i >= n || j < 0 || j >= n) {
        std::cerr << "Índice fuera de rango." << std::endl;
        return -1.0f;
    }
    float result;
    CHECK_CUDA_ERRORS(cudaMemcpy(&result, &distMatrix_device[i * n + j], sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

void Dataset::savePointFile() const{
    ofstream point_file;
    point_file.open("data/save/points_"+std::to_string(X.size())+".txt");
    
    for (size_t i = 0; i < X.size(); i++)
    {
        point_file << X[i] << "," << Y[i] << endl;
    }
    point_file.close();    
}