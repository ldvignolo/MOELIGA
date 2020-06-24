

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stack>
#include <ctime>
#include <fstream>
#include <string>

using namespace std;

/*================================
  FUNCIONES Y DEFINICIONES UTILES
  ================================*/


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



// Case Insensitive String Comparision 
bool compareChar(char & c1, char & c2)
{
    if (c1 == c2)
        return true;
    else if (std::toupper(c1) == std::toupper(c2))
        return true;
    return false;
}
bool caseInSensStringCompare(std::string & str1, std::string &str2)
{
    return ( (str1.size() == str2.size() ) &&
             std::equal(str1.begin(), str1.end(), str2.begin(), &compareChar) );
}


stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << endl;
    tictoc_stack.pop();
}


stack<clock_t> global_tictoc_stack;

void global_tic() {
    global_tictoc_stack.push(clock());
}

double global_toc() {
    
    double elapsed;
    
    cout << "TOTAL Time elapsed: "
              << ((double)(clock() - global_tictoc_stack.top())) / CLOCKS_PER_SEC
              << endl;
    
    elapsed = ((double)(clock() - global_tictoc_stack.top())) / CLOCKS_PER_SEC;
              
    global_tictoc_stack.pop();
    
    return elapsed;
}



