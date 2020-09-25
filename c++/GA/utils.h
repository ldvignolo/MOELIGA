

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



#include <chrono>

class newTimer
{
public:
    newTimer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};


newTimer local_timer; 
newTimer global_timer; 


void tic() {
    local_timer.reset();    
}

void toc() {
    cout << "Time elapsed: " << local_timer.elapsed() << endl;    
}

double toc(bool verbose) {
    
    double elapsed = local_timer.elapsed();
    if (verbose) {
        cout << "Time elapsed: "
                << elapsed
                << endl;
    }              
    return elapsed;
}


double toc2() {
    
    double elapsed = local_timer.elapsed();
    return elapsed;
}


void global_tic() {
    global_timer.reset();
}

double global_toc() {
    
    double elapsed = global_timer.elapsed();
    
    cout << "TOTAL Time elapsed: " << elapsed << endl;
    
    return elapsed;
}

double global_toc(bool verbose) {
    
    double elapsed = global_timer.elapsed();
    
    if (verbose) {
        cout << "TOTAL Time elapsed: " << elapsed << endl;
    }          
   
    return elapsed;
}



/* una función para convertir interos a cadena de caracteres */

char* itoa(int val, int base)
{
    static char buf[32] = {0};

    int i = 30;

    for(; val && i ; --i, val /= base)
       buf[i] = "0123456789abcdef"[val % base];

    return &buf[i+1];

}

template <typename T> string tostr(const T& t)
{ 
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

bool caseInSensStringCompare(std::string & str1, std::string str2)
{
    return ( (str1.size() == str2.size() ) &&
             std::equal(str1.begin(), str1.end(), str2.begin(), &compareChar) );
}

vector <string> SplitWords(string strString)
{
    int ws=-1,we=-5;
    unsigned int i = 0;
    vector <string> words;
    bool wd=false;
  
    // Skip over spaces at the beginning of the word
    while(isspace(strString.at(i)))
      i++;
    strString+='.';
  
    while(i < strString.length())
    {    
       if((isspace(strString.at(i)))||(ispunct(strString.at(i))))
       {      
           if ((wd)&&(we>=ws)){
               words.push_back(strString.substr(ws,we-ws+1));
               wd = false;
           }
       }else{
           if (!wd){
               ws = i; 
               we = i;
               wd=true;
           } else we=i;        
       }
       i++ ;
    }
    
    return words;
}


















/*
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

double toc(bool verbose) {
    
    double elapsed = ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    if (verbose) {
        cout << "Time elapsed: "
                << elapsed
                << endl;
    }          
    tictoc_stack.pop();
    return elapsed;
}


double toc2() {
    
    double elapsed =  ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    tictoc_stack.pop();
    return elapsed;
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
              
    // global_tictoc_stack.pop();  comento para llamarlo en cada generacion con referencia al mismo push
    
    return elapsed;
}

double global_toc(bool verbose) {
    
    double elapsed;
    
    if (verbose) {
        cout << "TOTAL Time elapsed: "
                << ((double)(clock() - global_tictoc_stack.top())) / CLOCKS_PER_SEC
                << endl;
    }          
    
    elapsed = ((double)(clock() - global_tictoc_stack.top())) / CLOCKS_PER_SEC;
              
    // global_tictoc_stack.pop();  comento para llamarlo en cada generacion con referencia al mismo push
    
    return elapsed;
}
*/
