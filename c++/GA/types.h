#ifndef _TYPES_
#define _TYPES_


#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

typedef bool gen;
typedef vector < gen > cromosoma;
typedef vector < double > fit_vect;



class individuo
{
      public:
          cromosoma crom;
          vector <short> index;  // para guardar referencia de INDIVIDUOS al pasar desde y hacia subpoblaciones
          bool evaluate;
          float x;
          unsigned rango; // moga rank
          double ncount;  // moga niche count
          vector < double > distancias; // moga
          fit_vect aptitud;    // moga
          double Fitness;      // moga fitness
          double sFitness;     // moga shared fitness
          int nr;              // moga nro ind con = rango
          int nF;              // number of selected features
          int padre1, padre2;
};


class resultado
{
      public:
              float x;
              float f;
              double maxfitness;
              double minfitness;
              double prom;
              vector < double > maxApts;   // moga              
};

class poblacion
{
      public:
             vector <individuo> individuos;
             int NMutas;
             int lcrom;
             double mean_dist;                  // distancia media entre los indidividuos
             short tampob;                      // NUMERO DE INDIVIDUOS EN LA POBLACION
             short Current_Front_Size;

};




/* una función para convertir interos a cadena de caracteres */

char* itoa(int val, int base){

	static char buf[32] = {0};

	int i = 30;
	
	for(; val && i ; --i, val /= base)
	
		buf[i] = "0123456789abcdef"[val % base];
	
	return &buf[i+1];
	
}

template <typename T> string tostr(const T& t) { 
   ostringstream os; 
   os<<t; 
   return os.str(); 
} 



#endif
