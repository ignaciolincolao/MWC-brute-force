// brute_force_algorithm.h: archivo de inclusión para archivos de inclusión estándar del sistema,
// o archivos de inclusión específicos de un proyecto.
#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <map>
#include <math.h>
#include <iostream>
#include <float.h>
#include <nlohmann/json.hpp>
#include <tuple>

using namespace std;
bool vector_sort(int const& lvd, int const& rvd);
// Function to calculate the distance between two points
double euclidian_distance(double x1, double y1, double x2, double y2);
// Function to evaluate the solutions and return the fitness value
double evaluate_solution(vector<int> pos, double** mat, int length);
tuple<double, int*> comb(int N, int K, double** distance_matrix);
//Solution from https://rosettacode.org/wiki/Combinations#C.2B.2B

#endif
