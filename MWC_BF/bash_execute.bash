#!/bin/bash

# Compilado de C++

cd build
compilado="./brute_force_algorithm"
datos="../data/test/40/"


# Leer cada archivo en el directorio
for archivo in "$datos"/*; do
  if [[ -f "$archivo" ]]; then
    # Extraer los valores del nombre del archivo
    IFS="_" read -r nombre puntos puntos_izq puntos_der seed <<< "$archivo"
    # Ejecutar el compilado de C++ con los parámetros
    "$compilado" "$archivo" "$puntos" "$puntos_izq" "$puntos_der" "$seed"
  fi
done
