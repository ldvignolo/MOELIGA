#ifndef __MEASURES_H__
#define __MEASURES_H__

# include <stdio.h>
//# include <stdlib.h>

# include <cstring>

# include <cstdlib>
# include <iostream>
# include <fstream>

//# include <time.h>

# include <vector>
# include <map>
# include <algorithm>

// # include "json-master/src/json.hpp"  // GITHUB ---> https://github.com/nlohmann/json
# include "json.hpp"  // GITHUB ---> https://github.com/nlohmann/json

# include "dictionary.hpp"
# include "../GA/types.h"


#include <chrono>
#include <ctime>



using jsonlib = nlohmann::json;


//================================================================ 
class Measures
{
    private:
        
        //========================================================
        // TODAS LAS MEDIDAS SE GUARDAN COMO DOUBLE POR DEFECTO
        //========================================================
        
        std::chrono::time_point<std::chrono::system_clock> start;
        
        std::chrono::time_point<std::chrono::system_clock> end;
        
        int Gmax;                                                                               // NUMERO MAXIMO DE GENERACIONES
        
        int G;                                                                                  // CONTADOR DE GENERACIONES
        
        int idx_elite;                                                                          // INDICE DE INDIVIDUO ELITE EN LA GENERACION ACTUAL
        
        std::map< std::string, std::vector<double> > elite;                                     // < measure, values > ELITE INDIVIDUAL
        
        std::map< std::string, std::map< std::string, std::vector<double> > > measures;         // < measure, < statistic, values > > MEASURES AND STATISTICS

    public:
        
        
        /* CONSTRUCTOR */
        Measures(int Gmax);
        
        
        /* SET METHOD */
        void Set(std::string measure);
        
        
        /* GET METHOD [HISTORY] -- VECTOR OF DOUBLES */
        std::vector<double> Get(std::string measure, std::string statistic);
        
        
        /* GET METHOD [HISTORY] -- DOUBLE */
        double Get(std::string measure, std::string statistic, int G);
        
        
        /* GET METHOD [ELITE INDIVIDUAL] -- VECTOR OF DOUBLES */
        std::vector<double> GetElite(std::string measure);
        
        
        /* GET METHOD [ELITE INDIVIDUAL] -- DOUBLE */
        double GetElite(std::string measure, int G);
        
        
        /* UPDATE METHOD -- GLOBAL */
        void Update(poblacion &P, int G);
        
        
        /* UPDATE METHOD -- VECTOR */
        void Update(std::string measure, int G, std::vector<double> values);
        
        
        /* SAVE MEASURES */
        void Save(const std::string FILENAME, Dictionary &SETTINGS);
    
};



//==========================================================

#endif 