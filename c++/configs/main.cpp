/*
 * =====================================================================================
 *
 *       Filename:  NSEAv01_r05.cpp
 *
 *    Description:  Algoritmo evolutivo basado en conjuntos anidados.
 *
 *        Version:  1.0
 *        Created:  10/04/13 11:35:00
 *       Revision:  5
 *       Compiler:  gcc
 *
 *         Author:  Matias Gerard.
 *   Organization:
 *
 * =====================================================================================
 */



# include <stdio.h>
# include <iostream>
# include <fstream>
# include <string>

# include <cstring>

# include <vector>
# include <map>


# include "Toolbox.hpp"   // LECTURA DEL ARCHIVO DE CONFIGURACION


// using namespace std;

//=============================================================================================



int main()//(int argc, char ** argv)
{
    std::vector<double> V(5,0);
    V[0] = 10;
    V[1] = 20;
    V[2] = 30;
    V[3] = 40;
    V[4] = 50;
    
    std::vector<double> W(4,0);
    W[0] = 1;
    W[1] = 5;
    W[2] = 10;
    W[3] = 100;
    
    Measures MEASURES(10);
    
    MEASURES.Update("Fitness", 0, V);
    MEASURES.Update("Fitness", 1, V);
    MEASURES.Update("Fitness", 2, V);
    MEASURES.Update("Fitness", 3, V);
    
    
    std::cout << "Fitness (max) [0]: " << MEASURES.Get("Fitness", "max", 0) << std::endl;
    std::cout << "Fitness (min) [0]: " << MEASURES.Get("Fitness", "min", 0) << std::endl;
    std::cout << "Fitness (mean) [0]: " << MEASURES.Get("Fitness", "mean", 0) << std::endl;
    std::cout << "Fitness (std) [0]: " << MEASURES.Get("Fitness", "std", 0) << std::endl;
    std::cout << "Fitness (median) [0]: " << MEASURES.Get("Fitness", "median", 0) << std::endl;
    std::cout << "Fitness (mad) [0]: " << MEASURES.Get("Fitness", "mad", 0) << std::endl;
    
    
    MEASURES.Update("Nfeatures", 0, W);
    MEASURES.Update("Nfeatures", 1, W);
    MEASURES.Update("Nfeatures", 2, W);
    MEASURES.Update("Nfeatures", 3, W);
    
    
    std::cout << "Nfeatures (max) [0]: " << MEASURES.Get("Nfeatures", "max", 0) << std::endl;
    std::cout << "Nfeatures (min) [0]: " << MEASURES.Get("Nfeatures", "min", 0) << std::endl;
    std::cout << "Nfeatures (mean) [0]: " << MEASURES.Get("Nfeatures", "mean",0) << std::endl;
    std::cout << "Nfeatures (std) [0]: " << MEASURES.Get("Nfeatures", "std", 0) << std::endl;
    std::cout << "Nfeatures (median) [0]: " << MEASURES.Get("Nfeatures", "median", 0) << std::endl;
    std::cout << "Nfeatures (mad) [0]: " << MEASURES.Get("Nfeatures", "mad", 0) << std::endl;
    
    //==============================================================================
    
    Dictionary SETTINGS("SETTINGS.cfg");
    
    std::cout << "SELECCION_str: " << SETTINGS.get_str("OpSeleccion") << std::endl;
    
    std::cout << "SELECCION_int: " << SETTINGS.get_int("OpSeleccion") << std::endl;
    
    std::cout << "M_str: " << SETTINGS.get_str("Nindividuos") << std::endl;
    
    std::cout << "M_int: " << SETTINGS.get_int("Nindividuos") << std::endl;
    
    
    MEASURES.Save("RESULTADOS.json", SETTINGS);
    
    return 0;
};