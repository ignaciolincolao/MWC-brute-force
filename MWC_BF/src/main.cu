#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cstdlib>
#include <random>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>


#include <Dataset.cuh>
#include <Coalition.cuh>

using namespace std;


int main(int argc, char* argv[]){
    if (argc < 5) {
        std::cout << "Faltan argumentos. Uso: programa <nombre> <puntos> <puntos_izq> <puntos_der>\n";
        return 1;
    }

    std::string nombre = argv[1];
    int puntos = std::stoi(argv[2]);
    int puntos_izq = std::stoi(argv[3]);
    int puntos_der = std::stoi(argv[4]);
    std::string mseed = argv[5];

    cout << mseed << endl;
    Dataset DATOS(nombre);
    //Dataset DATOS("data/test/points_100.txt");
    //Dataset DATOS("data/test/points_40.txt");
    //Dataset DATOS("data/votes.json","JSON");
    DATOS.printXY();
    cout << DATOS.getDistanciaHost(0,1) << " - " << DATOS.getDistanciaDevice(0,1) << endl;
    int quorum = trunc(DATOS.X.size() / 2)+1;

    Coalition COALITION(quorum,DATOS.X.size(),DATOS.distMatrix_device,1024,1024); 
    //Coalition COALITION(quorum,DATOS.X.size(),DATOS.distMatrix_device,5078,1024); 
    // Time variable initialization for execution calculation
    auto initial_time = chrono::high_resolution_clock::now();
    cout << "Generando combinatoria!" << endl;
    COALITION.BestSolution();

    auto final_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(final_time - initial_time).count();



    time_taken *= 1e-9;
    cout << "Time:"<< fixed << time_taken << setprecision(9) << endl;
    cout << "Minimum Fitness:" << COALITION.bestFitness << endl;
    cout << "Coalition:" << endl;
    for (size_t i = 0; i < quorum; i++)
    {
        cout << COALITION.bestSolution[i] << ",";
    }


    ofstream result_file;
    result_file.open("../data/result/"+std::to_string(puntos)+"_"+std::to_string(puntos_izq)+"_"+std::to_string(puntos_der)+"_"+"_seed_"+mseed+"_result_"+ std::to_string(DATOS.X.size() )+"_"+std::to_string(time(0))+"p.txt");
    result_file << "Time:"<< fixed << time_taken << setprecision(9) << endl;
    result_file << "Minimum Fitness:" << COALITION.bestFitness << endl;
    result_file << "N_BLOCK: " << COALITION.nBlock << " N_THREADS:" << COALITION.nThread << endl;
    result_file << "Coalition:" << endl;
    for (size_t i = 0; i < quorum; i++)
    {
        result_file << COALITION.bestSolution[i] << ",";
    }
    result_file.close();  


    return 0;
}