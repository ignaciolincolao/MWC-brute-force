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


int main() {

    //Dataset DATOS("data/test/Dataset_25_12_13_seed-7.csv");
    //Dataset DATOS("data/test/points_100.txt");
    Dataset DATOS("data/test/points_40.txt");
    //Dataset DATOS("data/votes.json","JSON");
    DATOS.printXY();
    cout << DATOS.getDistanciaHost(0,1) << " - " << DATOS.getDistanciaDevice(0,1) << endl;
    int quorum = trunc(DATOS.X.size() / 2)+1;

    Coalition COALITION(quorum,DATOS.X.size(),DATOS.distMatrix_device,32,1024); 

    // Time variable initialization for execution calculation
    auto initial_time = chrono::high_resolution_clock::now();
    cout << "Generando combinatoria!" << endl;
    COALITION.BestSolution();
    // Stop the clock
    auto final_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(final_time - initial_time).count();

    // Convert the time taken by the algorithm to seconds


    time_taken *= 1e-9;
    cout << "Time:"<< fixed << time_taken << setprecision(9) << endl;
    cout << "Minimum Fitness:" << COALITION.bestFitness << endl;
    cout << "Coalition:" << endl;
    for (size_t i = 0; i < quorum; i++)
    {
        cout << COALITION.bestSolution[i] << ",";
    }


    ofstream result_file;
    result_file.open("../data/result/result_"+ std::to_string(DATOS.X.size() )+".txt");
    result_file << "Time:"<< fixed << time_taken << setprecision(9) << endl;
    result_file << "Minimum Fitness:" << COALITION.bestFitness << endl;
    result_file << "Coalition:" << endl;
    for (size_t i = 0; i < quorum; i++)
    {
        result_file << COALITION.bestSolution[i] << ",";
    }
    result_file.close();  


    return 0;
}