#include "brute_force_algorithm.h"
#include <chrono>
#define pb push_back
#include <vector>

using namespace std;
using json = nlohmann::json;

int main() {
    int n = 25;
    float new_data[n][2];
    std::ifstream file("Dataset_445_222_223_seed-2.csv");
    std::string   line;
    int count = 0;
    while(std::getline(file, line) && count < n)
    {
        std::stringstream   linestream(line);
        std::string         data;
        float                 val1;
        float                 val2;
        char                  coma;
        linestream >> val1 >> coma >> val2;
        new_data[count][0] = val1;
        new_data[count][1] = val2;
        cout << new_data[count][0] << "-"<< new_data[count][1] << endl;
        count++;
    }

    // Call the Json file
    /*
    ifstream file("votes.json");
    json data = json::parse(file);
    */
    //int n = 25;//data["rollcalls"][0]["votes"].size();
    int quorum = trunc(n / 2)+1;
    // Initialize the distance matrix
    double** distance_matrix = (double**)malloc(n * sizeof(double*));
    // Crea archivo
    /*
    ofstream point_file;
    point_file.open("points.txt");
    
    for (size_t i = 0; i < n; i++)
    {
        point_file << data["rollcalls"][0]["votes"][i]["x"] << "," << data["rollcalls"][0]["votes"][i]["y"] << endl;
    }
    point_file.close();    
    */
   


    for (size_t i = 0; i < n; i++)
    {
        distance_matrix[i] = (double*)malloc(n * sizeof(double));
    }


    // Fill the distance matrix
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            //distance_matrix[i][j] = euclidian_distance(data["rollcalls"][0]["votes"][i]["x"], data["rollcalls"][0]["votes"][i]["y"], data["rollcalls"][0]["votes"][j]["x"], data["rollcalls"][0]["votes"][j]["y"]);
            distance_matrix[i][j] = euclidian_distance(new_data[i][0], new_data[i][1], new_data[j][0], new_data[j][1]);
        }
    }
    // Time variable initialization for execution calculation
    auto initial_time = chrono::high_resolution_clock::now();
    cout << "Generando combinatoria!" << endl;
    double fitness;
    int* coalition = (int*)malloc(quorum * sizeof(int));
    tie(fitness,coalition)=comb(n, quorum,distance_matrix);
    cout << "Fin Combinatoria!" << endl;
    // Stop the clock
    auto final_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(final_time - initial_time).count();
    // Convert the time taken by the algorithm to seconds
    time_taken *= 1e-9;
    cout << "Time:"<< fixed << time_taken << setprecision(9) << endl;
    cout << "Minimum Fitness:" << fitness << endl;
    cout << "Coalition:" << endl;
    for (size_t i = 0; i < quorum; i++)
    {
        cout << coalition[i] << ",";
    }


    ofstream result_file;
    result_file.open("result_"+ std::to_string(n)+".txt");
    result_file << "Time:"<< fixed << time_taken << setprecision(9) << endl;
    result_file << "Minimum Fitness:" << fitness << endl;
    result_file << "Coalition:" << endl;
    for (size_t i = 0; i < quorum; i++)
    {
        result_file << coalition[i] << ",";
    }
    result_file.close();   
    return 0;
}