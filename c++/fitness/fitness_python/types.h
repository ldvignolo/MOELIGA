#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

typedef bool gen;
typedef vector < gen > cromosoma;

class individuo
{
      public:
              cromosoma crom;
	      vector <short> index;
              float x;
              double aptitud;
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
};

typedef vector <individuo> poblacion;

/* una función para convertir interos a cadena de caracteres */

char* itoa(int val, int base){

	static char buf[32] = {0};

	int i = 30;
	
	for(; val && i ; --i, val /= base)
	
		buf[i] = "0123456789abcdef"[val % base];
	
	return &buf[i+1];
	
}
