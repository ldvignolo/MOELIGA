//----------------------------------------------------------------------------
//
//  Algoritmo Genético
//
//  Versión Paralela
//
//  29/05/07 - ...
// 
//  26-02-13 - Adaptacion para el challenge Interspeech ComParE
//
// 2016-01-05 - Seleccion de caracteristicas / subpoblaciones
// 2016-03-31 - Tasa de Mutacion con decaimiento Exponencial
//            - La tasa de mutacion de subpoblacion se actualiza proporcinalmente segun el largo del cromosoma
// 
// TODO:
//    - tasa de mutacion fuzzy adaptativa
//
//  Leandr0
//
//----------------------------------------------------------------------------


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <float.h>
#include <vector>
#include <time.h>
#include <algorithm>
#include <math.h> 
#include <random>

#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR

#include <mpi.h>

#include "types.h"
#include "utils.h"

#include <stack>
#include <ctime>

#include "../configs/Toolbox.hpp"   // LECTURA DEL ARCHIVO DE CONFIGURACION



using namespace std;

const int maxpob = 20000;
const int maxstring = 300;



/*==================================
  DEFINICION DEL ALGORITMO GENETICO
  ==================================*/

class AG {

private:
       double pcruza, pmutacion;               // PROBABILIDADES DE CRUZA Y MUTACION
       double prom, max, min;                  // GUARDO ESTADISTICAS SOBRE FITNESS PARA MOSTRAR
       MPI_Comm everyone;                      // VARIABLE DE MPI       
       fit_vect aptitud_min, aptitud_max;      // minimo y maximo historico observado para cada objetivo
       std::ofstream results;                  // ARCHIVO DE TEXTO DE RESULTADOS
       std::default_random_engine generator;       
       vector <vector<short>> clusters;        // array con los clusters actuales: cada fila tiene los indices de un cluster

public:
  
       int nObjctvs;                           // CANTIDAD DE OBJETIVOS  (no uso short porque se manda por mpi como int)
       short maxgen;                           // NUMERO MAXIMO DE GENERACIONES
       short gen;                              // GENERACION ACTUAL              
       short nsubpob;                          // CANTIDAD DE SUBPOBLACIONES       
       unsigned Elite;                            
       short dist_opt;                         // medida de distancia para fitness sharing: 0-'variable space', 1-'objective space'
       double alfa;                            
       double sigma_share, alfa_share;         
       string string_aux_filename = "";            
       float activ_rate, activ_rate_sp;        // TASA DE ACTIVACIONES PARA INICALIZACION
       bool stepped_activ;                     // tasa de activacion variable / escalonada (true) o fija (false) / solo para la poblacion general
       vector <float> activ_rates;             // vector tasas de activacion para el caso escalonado
       bool ModifyRepeated;                    // bandera para aplicar operador de variacion en cromosomas repetidos     
       short FitnessOption;                    // opcion para elegir el esquema de Fitness
       bool FitnessNRScale;                    // multiplicar Fitness por nro de indiv en el mismo rank
       short Rfun;                             // medida de ranking para elegir el mejor individuo del ultimo frente 
                                               
       poblacion pobvieja;                     // POBLACION EN LA GENERACION ACTUAL
       poblacion pobnueva;                     // POBLACION DE LA SIGUIENTE GENERACION
                                               
       string fecha;                           // FECHA ACTUAL
       string slvbin;                          // Nombre del ejecutvable fitness
       string res_file;                        // nombre de archivo TXT de resultados
       string yaml_file;                       // nombre de archivo yaml de resultados
       string crom_file;                       // nombre de archivo TXT de los mejores cromosomas (frente Pareto)
       string folder;                          // directorio para los archivos de resultados
       
       bool verbose=false;
       /*-----------------
             METODOS 
         -----------------*/
       
       // INICIALIZAR CROMOSOMA
       cromosoma initcrom(int in_lcrom, float activ_rate);

       // INICIALIZAR POBLACION
       void inicializar(int in_tampob, int in_lcrom, int in_maxgen, double in_pcruza, double in_pmutacion, int nproc, float tasa_activ, string SETTINGS);
       
       
       // DETERMINAR EL NUMERO DE INDIVIDUOS QUE SERAN SELECCIONADOS COMO PADRES???
       int seleccion(int tampob, float sumaptitud, poblacion *pob, int caso);
       
       
       // OPERADOR DE CRUZA
       void cruza(cromosoma padre1, cromosoma padre2, cromosoma *hijo1, cromosoma *hijo2, int Nlcrom, float pcruza);
       
       
       // OPERADOR DE MUTACION
       cromosoma mutacion(cromosoma crom, double pmutacion, int caso, int &nmutas);
       
       // INVERTIR bit
       bool flip(float prob);
       
       // GENERARAR LA SIGUIENTE POBLACION
       void generacion(int brecha, int seltype, int mutatype, double pmutacion, int nproc);
       
       
       // GENERARAR LA SIGUIENTE SUBPOBLACION
       poblacion generSubPob(poblacion &SubPob, int brecha, int seltype, int mutatype, float pmuta, int nproc, float imax, short subgen, short maxsubgen, string T_reemplazo, bool sp_c_eval);
       
       
       // GENERAR y EVOLUCIONAR SUBPOBLACION
       void EvoSubPobs(int brecha, int seltype, int mutatype, float pmuta, int nproc, int Nsubpobs, int tamSubPob, int NGenSubPob, string cfg_settings);
       
       
       // EXTRAER RESULTADOS DE LA POBLACION???
       statistics estadisticas(poblacion &inPOB);

       // calcular distancia entre individuos
       double distancia(individuo indiv_x, individuo indiv_y, int in_cromlen, short nObjctvs, short dist_opt);
       double distancia(individuo indiv_x, vector<double> centroide, int in_cromlen);
       

       // FUNCION DE FITNESS SHARING
       double sharing_fun(double dist, double sigma, double alfa);
       
       void CalcularFitness(poblacion &inPOB);

       // IMPRIMIR datos de generacion en TXT
       void ImprimirGen(int generac, double maxfitness, double minfitness, double prom, poblacion &pob);
       void yaml_ImprimirGen(int generac, double maxfitness, double minfitness, double prom, poblacion &pob);
       
       // IMPRIMIR datos de individuo en TXT
       // void Imprimir(double *fitness, int generac, int indiv, cromosoma genes);       
       void ImprimirCromo(individuo johndoe, int generac, int indiv, bool new_front);
       void yaml_ImprimirCromo(individuo johndoe, int generac, int indiv, bool new_front);
       void ImprimirCromoForTest(individuo johndoe, bool append);
       
       // Buscar mejor individuo e IMPRIMIR en TXT
       void ImprimirFrente(poblacion &inPOB, int generac, double max_fit, bool onlyBest, bool yaml);
       
       // FUNCION AUXILIAR PARA IMPRIMIR ALGUN TIPO DE NOTIFICACION EN EL ARCHIVO DE RESULTADOS
       void Notificar(string notify);
       
       unsigned Verificar(individuo *johndoe);       
       unsigned Verificar(cromosoma *johndoe);
       
       void CalcularDistancias(poblacion &inPOB, bool operador_diversidad);
       
       // DETENER EL ALGORITMO???
       void Terminar(int nproc, int lcrom, bool txt);
};



//==================================
// METODOS DE "MOGA"
//==================================


//===================================================================
statistics AG::estadisticas(poblacion &inPOB)
{
   int j;
   float acum;

   statistics result;

   acum = inPOB.individuos[0].Fitness;

   for (j=1;j<inPOB.tampob;j++)
   {
         if (inPOB.individuos[j].Fitness > acum)
         {
            acum = inPOB.individuos[j].Fitness;
         }
   }

   result.maxfitness = acum;

   acum = inPOB.individuos[0].Fitness;

   for (j=1;j<inPOB.tampob;j++)
   {
         if (inPOB.individuos[j].Fitness < acum)
         {
             acum = inPOB.individuos[j].Fitness;
         }
   }
   result.minfitness = acum;

   acum = inPOB.individuos[0].Fitness;

   for (j=1;j<inPOB.tampob;j++)
   {
         acum = acum + inPOB.individuos[j].Fitness;
   }
   result.prom = acum/inPOB.tampob;

   return result;

}
//===================================================================



//===================================================================
cromosoma AG::initcrom(int in_lcrom, float activ_rate)
{
    cromosoma adan;
    vector <int> idx;
    int nact = 0;
    
    adan.resize(in_lcrom);
    idx.resize(in_lcrom);
    for (int i=0;i<in_lcrom;i++) {
      adan[i]=false;
      idx[i]=i;
    }  
    
    random_shuffle ( idx.begin(), idx.end() );

    // std::lognormal_distribution<double> distribution(activ_rate,0.5);
    std::normal_distribution<double> distribution(10.0*activ_rate,2.5);
    double number = distribution(generator);
    
    nact = (int) ((number/10.0)*in_lcrom);
    
    if (flip(0.02)) nact = in_lcrom - (int) rand()%in_lcrom*10.0/100.0;
    else if (flip(0.02)) nact = (int) rand()%in_lcrom*10.0/100.0;
    
    if ((nact<0) || (nact>=in_lcrom))
       nact = (int) (activ_rate*in_lcrom);
    
    // nact = (int) (activ_rate*in_lcrom);
    // cout  << ">> -- >> -- >> -- >> " << activ_rate << " " << in_lcrom   <<  " "  <<   number    <<  " "  << nact << endl;
    
    for (int i=0;i<nact;i++) {
      adan[idx[i]]=true;
    }
    
    // cout  << ">> -- >> -- >> -- >> " << adan[0] << " " << adan[1] << " " << adan[2] << " " << adan[3]<< " " << adan[4]<< " " << adan[5]<< endl;
    
    return adan;
}
//===================================================================



//===================================================================
void AG::generacion(int brecha, int seltype, int mutatype, double in_pmutacion, int nproc)
{

    int i, k, j, mate1, mate2;
    unsigned aux;
    int punto;
    double sumaptitud=0;

    pmutacion = in_pmutacion;
    
    pobnueva.lcrom = pobvieja.lcrom;
    pobnueva.tampob = pobvieja.tampob;
    pobnueva.NMutas = 0;
    
    // Elitismo basado en rango / dominacion
    aux = pobvieja.individuos[0].rango;
    punto = 0;
    for (j=1;j<pobvieja.tampob;j++)
    {
        if (pobvieja.individuos[j].rango < aux)
        {
              aux = pobvieja.individuos[j].rango;
              punto = j;
        }
    }    
    
    vector <short> new_Elite;
    // En este caso se hace un muestreo de los individuos que estan en el frente
    for (j=0;j<pobvieja.tampob;j++)
    {      
        pobvieja.individuos[j].edad++;
        if (pobvieja.individuos[j].rango==pobvieja.individuos[punto].rango) 
        {
              new_Elite.push_back(j);
        }
    }    
    
    unsigned rango = pobvieja.individuos[punto].rango+1;
    while ((new_Elite.size() < Elite) && (rango<((unsigned) pobvieja.tampob)))
    {       
        for (j=0;(j<pobvieja.tampob)&&(new_Elite.size()<Elite);j++)
        {      
             if (pobvieja.individuos[j].rango==rango) 
             {
                  new_Elite.push_back(j);
             }
         }    
         rango++;
    }     
     
    
    // while (new_Elite.size() > Elite)
    while (new_Elite.size() > ((unsigned) pobvieja.tampob))         
    {
        double min_dist = DBL_MAX;
        int jmin=0;
        for (size_t ii=0;ii<new_Elite.size();ii++)
            for (size_t jj=0;jj<new_Elite.size();jj++)
                if ((ii!=jj) && (min_dist < pobvieja.individuos[new_Elite[ii]].distancias[new_Elite[jj]])) {
                    min_dist = pobvieja.individuos[new_Elite[ii]].distancias[new_Elite[jj]];
                    jmin = jj;
                }     
                
        new_Elite.erase(new_Elite.begin()+jmin);          
    }    
    short nE = new_Elite.size();
    
    for (i=0;i<nE;i++)
       pobnueva.individuos[i] = pobvieja.individuos[new_Elite[i]];

    
    //-----------------------------------------------------------------------------//

    tic();

    j = nE+brecha;
    
    if (j > pobvieja.tampob) j = pobvieja.tampob;
    
    // brecha generacional
    for (i=nE; i<j; i++)
        pobnueva.individuos[i] = pobvieja.individuos[seleccion(pobvieja.tampob, sumaptitud, &pobvieja, seltype)];
 
    cromosoma newbie1, newbie2;
    
    while ( j < pobvieja.tampob )
    {
        mate1 = seleccion(pobvieja.tampob, sumaptitud, &pobvieja, seltype);
        mate2 = seleccion(pobvieja.tampob, sumaptitud, &pobvieja, seltype);

        newbie1 = mutacion(pobvieja.individuos[mate1].crom, pmutacion, mutatype, pobnueva.NMutas);
        newbie2 = mutacion(pobvieja.individuos[mate2].crom, pmutacion, mutatype, pobnueva.NMutas);
    
        if (j<(pobvieja.tampob-1)) cruza(newbie1, newbie2, &pobnueva.individuos[j].crom, &pobnueva.individuos[j+1].crom, pobvieja.lcrom, pcruza);
        else  cruza(newbie1, newbie2, &pobnueva.individuos[j].crom, &pobnueva.individuos[j].crom, pobvieja.lcrom, pcruza);

        pobnueva.individuos[j].padre1 = mate1;
        pobnueva.individuos[j].padre2 = mate2;
        pobnueva.individuos[j].edad = 0;

        if (j<(pobvieja.tampob-1))
        {
            pobnueva.individuos[j+1].padre1 = mate2;
            pobnueva.individuos[j+1].padre2 = mate1;
            pobnueva.individuos[j+1].edad = 0;
        }

        j = j + 2;

    } 
    
    CalcularDistancias(pobnueva, ModifyRepeated);
   
    /*------ MPI VARIOS + Var Paralelizacion ------*/

    MPI_Request request[nproc];
    MPI_Status status;
    int buffer[pobvieja.lcrom];
    int flag;
    bool busys[nproc];
    double aptitud[nproc][nObjctvs];
    double *pointApt;
    int map[nproc];
    int tag;
    int params[3];

    /*---------------------------------------------*/

    params[0] = pobvieja.lcrom;  
    params[1] = 0;               // 0 indica pob principal; 1 indica subpob
    params[2] = 1;               // indica si se debe usar el clasificador para evaluar
    
   // repartir calculos de aptitud entre nodos
    
    for (i=0;i<nproc;i++) busys[i] = false;
    tag = 1000 + nproc;
    j = Elite+brecha; // los valores anteriores de los objetivos sirven, lo que hay que recalcular es el FITNESS
    while (j<pobvieja.tampob)
    {
       Verificar(&pobnueva.individuos[j]);
       for (i=0;((i<nproc)&&(j<pobvieja.tampob));i++)
       {
           if  (!busys[i])
           {
               map[i]=j;
               MPI_Send(params, 3, MPI_INTEGER, i, tag, everyone);
               for (k=0;k<pobvieja.lcrom;k++)
                  if (pobnueva.individuos[j].crom[k]) buffer[k] = 1; else buffer[k] = 0;
               MPI_Send(buffer, pobvieja.lcrom, MPI_INTEGER, i, tag, everyone);
      
               busys[i] = true;
               pointApt = &(aptitud[i][0]);
               MPI_Irecv(pointApt, nObjctvs, MPI_DOUBLE, i, tag, everyone, &request[i]);
               j++;
           }
       }

       for (i=0;i<nproc;i++)
       {
           flag = 0;
           if (busys[i]) MPI_Test(&request[i],&flag,&status);
           if (flag == 1)
           {
               busys[i] = false;
               
               // for (k=0;k<nObjctvs;k++) pobnueva.individuos[map[i]].aptitud[k] = aptitud[i][k];    // MOGA               
               for (k=0;k<nObjctvs;k++) {
                   pobnueva.individuos[map[i]].aptitud[k] = aptitud[i][k];  // MOGA
                   if (aptitud[i][k]>aptitud_max[k]) aptitud_max[k]=aptitud[i][k];
                   if (aptitud[i][k]<aptitud_min[k]) aptitud_min[k]=aptitud[i][k];
               }
               
           }
       }
       
    }

    for (i=0;i<nproc;i++)
    {
       if (busys[i])
       {
           flag = 0;
           while (flag!=1) MPI_Test(&request[i],&flag,&status);
           busys[i] = false;
           // for (k=0;k<nObjctvs;k++) pobnueva.individuos[map[i]].aptitud[k] = aptitud[i][k];    // MOGA
           for (k=0;k<nObjctvs;k++) {
               pobnueva.individuos[map[i]].aptitud[k] = aptitud[i][k];  // MOGA
               if (aptitud[i][k]>aptitud_max[k]) aptitud_max[k]=aptitud[i][k];
               if (aptitud[i][k]<aptitud_min[k]) aptitud_min[k]=aptitud[i][k];
           }
       }
    }
  
    CalcularFitness(pobnueva);
    
    double caux=0.0;
    for (size_t ii=0;ii<clusters.size();ii++) caux=caux+clusters[ii].size();
    caux=caux/clusters.size();    
    double fit_aux=0.0; int ibest=0;
    for (i=0;i<pobnueva.tampob;i++)
        if (fit_aux<=pobnueva.individuos[i].Fitness)
        {
            fit_aux=pobnueva.individuos[i].Fitness;
            ibest=i;
        }

    if (verbose) {
        cout << "Generacion " << gen << " - Obj1: " << pobnueva.individuos[ibest].aptitud[0] << ", Obj2: " << pobnueva.individuos[ibest].aptitud[1] << ", Fsize: " << pobnueva.Current_Front_Size; 
        cout << ", Clusters: " << clusters.size() << ", AvgClusSize: " << (int) caux << ", MeanDist: " << pobnueva.mean_dist << endl;
    }
    
    double elapsed = toc(verbose);
    (void) elapsed;
    
    return;

}
//===================================================================




double AG::distancia(individuo indiv_x, individuo indiv_y, int in_cromlen, short nObjctvs, short dist_opt)
{

    double aux, dist = 0.0;
    
    if (dist_opt==0) 
    {
        for (int r=0;r<in_cromlen;r++) {
            // if (indiv_x.crom[r]!=indiv_y.crom[r]) dist=dist+1.0;     
            dist = dist + abs(indiv_x.crom[r]-indiv_y.crom[r]);      
        }    
        // dist = sqrt(dist/in_cromlen);  
        dist = (dist/in_cromlen); 
       
    } else if (dist_opt==1) 
    {
        for (int r=0;r<nObjctvs;r++)
        {
            // aux = (indiv_x.aptitud[r]-indiv_y.aptitud[r]) / (aptitud_max[r]-aptitud_min[r]);
            aux = (indiv_x.aptitud[r]-indiv_y.aptitud[r]); 
            dist = dist + fabs(aux);
        }
        // dist = sqrt(dist/nObjctvs);
        dist = (dist/nObjctvs);
        
    } else 
    {         
        dist = 1.0;
    } 
    
    return dist;
}            



double AG::distancia(individuo indiv_x, vector<double> centroide, int in_cromlen)
{

    double dist = 0.0;

    for (int r=0;r<in_cromlen;r++) 
        dist = dist + pow((indiv_x.crom[r] - centroide[r]),2.0);
    
    dist = sqrt(dist/in_cromlen);  

    return dist;
}      





void AG::CalcularDistancias(poblacion &inPOB, bool operador_diversidad)
{
    int i, j, k;
    double distance;   
    float flip_rate = 0.1;

    vector <int> idx;
    int nch = 0;
    idx.resize(inPOB.lcrom);
    for (i=0;i<inPOB.lcrom;i++) idx[i]=i;                     
 
    if (inPOB.mean_dist<0.1) flip_rate = 10*flip_rate;
    
    inPOB.mean_dist = 0.0;
    for (j=0;j<inPOB.tampob;j++)
    {   
        inPOB.individuos[j].distancias.clear();
        inPOB.individuos[j].distancias.resize(inPOB.tampob);
        
        inPOB.individuos[j].distancias[j]=0.0;
        for (i=0;i<j;i++)
        {    
            // calcular las distancia de cada solucion i a j            
            distance = distancia(inPOB.individuos[i],inPOB.individuos[j],inPOB.lcrom,nObjctvs,dist_opt);
            
            if ((distance<=1e-10)&&(operador_diversidad))
            {
                random_shuffle ( idx.begin(), idx.end() );                
                nch = ceil(inPOB.lcrom*flip_rate/100);
                if (nch==0) nch=1;
                for (k=0;k<nch;k++)
                inPOB.individuos[j].crom[idx[k]] = !inPOB.individuos[j].crom[idx[k]];
                inPOB.individuos[j].edad = 0;
            }
            
            distance = distancia(inPOB.individuos[i],inPOB.individuos[j],inPOB.lcrom,nObjctvs,dist_opt);
            
            inPOB.individuos[j].distancias[i] = distance;
            inPOB.individuos[i].distancias[j] = distance;
            inPOB.mean_dist = inPOB.mean_dist + distance;
        }
    }
    
    inPOB.mean_dist =  inPOB.mean_dist / (inPOB.tampob*(1.0*inPOB.tampob/2.0));

    return;    
}









void AG::CalcularFitness(poblacion &inPOB)
{
    // recalcular segun reglas del MOGA
    // calculo el rango r(x,t) para cada individuo
    int i, j, r;
    unsigned k;
    unsigned mxrango=0;
    double acum;    
    
    inPOB.histograma = std::vector<short>(inPOB.lcrom,0); // inicializo con ceros
    
    for (j=0;j<inPOB.tampob;j++)
    {
       inPOB.individuos[j].nF = 0;
       for (i=0;i<inPOB.lcrom;i++) {
           if (inPOB.individuos[j].crom[i]) {
               inPOB.individuos[j].nF++;
               inPOB.histograma[i]++; 
           }
       }    
                   
       inPOB.individuos[j].R1 = 1.0 - pow(pow((1.0 - inPOB.individuos[j].aptitud[0]),2.0) + pow((1.0 - inPOB.individuos[j].aptitud[1]),2.0), 0.5);
       inPOB.individuos[j].R2 = inPOB.individuos[j].aptitud[0]/inPOB.individuos[j].nF;              
    }    
    
    bool flag1, flag2;
    for (j=0;j<inPOB.tampob;j++)
    {   
        inPOB.individuos[j].rango = 1;

        for (i=0;i<inPOB.tampob;i++)
        {    
            // contar las soluciones q dominan a j
            flag1=true; flag2=false;
            if (i!=j) for (r=0;r<nObjctvs;r++) 
            {  
                if (inPOB.individuos[i].aptitud[r] < inPOB.individuos[j].aptitud[r]) { flag1=false; break; }
                if (inPOB.individuos[i].aptitud[r] > inPOB.individuos[j].aptitud[r]) flag2=true; 
            }
            if ((flag1) && (flag2)) inPOB.individuos[j].rango++;  // r(x,t)=1+nq(x,t)
            
            if (inPOB.individuos[j].rango>mxrango) mxrango = inPOB.individuos[j].rango;
            
        }

    }
    
    /* ------------------------------------------------------------------
    * ------------ identificar clusters de individuos -------------------
    * ---------------------------------------------------------------- */
    
    bool clus_flag1=false;
    for (size_t r=0; r<clusters.size();r++) clusters[r].clear();
    clusters.clear();    
    
    for (i=0;i<inPOB.tampob;i++)
    {           
        
        for (size_t r=0; r<clusters.size();r++)
        {
            if (!(std::find(clusters[r].begin(), clusters[r].end(), i) != clusters[r].end()))  // true -> i NO esta en clusters[r]
            {
                clus_flag1=true;
                for (size_t s=0; s<clusters[r].size();s++)
                    if (0.0==sharing_fun(inPOB.individuos[i].distancias[clusters[r][s]], sigma_share, alfa_share)) clus_flag1=false;
                if ((clusters[r].size()>0)&&(clus_flag1)) clusters[r].push_back(i);                    
            }
                
        }
        clus_flag1=false;
        
        for (j=0;j<i;j++)
        {                        
            if (0.0<sharing_fun(inPOB.individuos[i].distancias[j], sigma_share, alfa_share))  // sharing>0 <=> distancia<sigma
            {    
                for (size_t r=0; r<clusters.size();r++)
                {
                    clus_flag1=false;
                    if ( std::find(clusters[r].begin(), clusters[r].end(), i) != clusters[r].end() )  // true -> i esta en clusters[r]
                    {
                        if ( !(std::find(clusters[r].begin(), clusters[r].end(), j) != clusters[r].end()) )  // true -> j NO esta en clusters[r]
                        {
                            clus_flag1=true;
                            for (size_t s=0; s<clusters[r].size();s++) // recorro los elementos del cluster r                                
                            {    
                                if (0.0==sharing_fun(inPOB.individuos[j].distancias[clusters[r][s]], sigma_share, alfa_share)) 
                                { 
                                    clus_flag1=false;
                                    break;
                                }    
                            }    
                            if (clus_flag1)   // si true -> j esta cerca de todos los elementos del cluster
                            { 
                                clusters[r].push_back(j);
                                break;
                            }    
                            
                        } else {              // implica que j ya esta en un cluster junto con i
                            clus_flag1=true;
                            break;            
                        }    
                    }
                    if (clus_flag1) break;    // implica que ya se agrego j en un cluster junto con i
                }
                if (!clus_flag1)              // salio del for r sin haber cluster con i y j
                {
                    vector <short> aux;
                    aux.push_back(i);
                    aux.push_back(j);
                    clusters.push_back(aux);
                }
                
            }

        }
    }

    /* ------------------------------------------------------------------
    * -------------------------------------------------------------------
    * ---------------------------------------------------------------- */

    vector <short> nk;
    // double ax;
    double axc;
    // calcular niche count y fitness
    for (j=0;j<inPOB.tampob;j++)
    {   
        inPOB.individuos[j].nr = 0;
        inPOB.individuos[j].ncount = 1;
    }
    
    for (j=0;j<inPOB.tampob;j++)
    {           
        inPOB.individuos[j].mean_dist = 0.0;
        for (i=0;i<inPOB.tampob;i++)
        {   
            // calcular niche count, y nr
            if ((inPOB.individuos[i].rango == inPOB.individuos[j].rango)&&(i!=j)) {
               inPOB.individuos[i].nr++; // numero de individuos que comparten el rango               
               acum = sharing_fun(inPOB.individuos[i].distancias[j], sigma_share, alfa_share);
               inPOB.individuos[i].ncount = inPOB.individuos[i].ncount + acum; // nc(x,t)               
               // inPOB.individuos[j].mean_dist += inPOB.individuos[j].distancias[i];               
            }            
            inPOB.individuos[j].mean_dist += inPOB.individuos[j].distancias[i];
        }
        inPOB.individuos[j].mean_dist = inPOB.individuos[j].mean_dist/inPOB.tampob;
    }
  
    nk.resize(mxrango);
    for (k=0;k<mxrango;k++)
    {
        nk[k] = 0;
        for (i=0;i<inPOB.tampob;i++) if (inPOB.individuos[i].rango == (k+1)){ nk[k]++; }          
    }
    
    for (j=0;j<inPOB.tampob;j++)
    { 
        // calcular fitness
        axc = 0;
        inPOB.individuos[j].Fitness = inPOB.tampob;
        for (k=0;k<(inPOB.individuos[j].rango-1);k++) 
        {            
            // ax = nk[k] - 0.5*(inPOB.individuos[j].nr - 1);
            // axc = axc + std::max(ax,0.0);            
            axc = axc + nk[k];
        }    
        inPOB.individuos[j].Fitness = inPOB.individuos[j].Fitness - axc - 0.5*(inPOB.individuos[j].nr - 1);
        // inPOB.individuos[j].Fitness = inPOB.individuos[j].Fitness - axc;
        // inPOB.individuos[j].Fitness = inPOB.individuos[j].Fitness - inPOB.individuos[j].rango;
        
        // calcular shared fitness        
        switch (FitnessOption)
        {        
            case 1:
                inPOB.individuos[j].sFitness = inPOB.individuos[j].Fitness / inPOB.individuos[j].ncount;
                break;        
            case 2:        
                if (inPOB.individuos[j].mean_dist>0.0)
                    inPOB.individuos[j].sFitness = inPOB.individuos[j].Fitness*inPOB.individuos[j].mean_dist/inPOB.individuos[j].nr;
                else
                    inPOB.individuos[j].sFitness = inPOB.individuos[j].Fitness;        
                break;                
            case 3:
                inPOB.individuos[j].sFitness = inPOB.individuos[j].Fitness;
                break;
        }
    }

    if (FitnessOption==3) 
    {    
        for (size_t jj=0;jj<clusters.size();jj++)
        {
            vector <double> centroide;
            for (i=0;i<inPOB.lcrom;i++)
            {
                double auxc=0.0;
                for (size_t kk=0;kk<clusters[jj].size();kk++)
                {
                    auxc = auxc + inPOB.individuos[clusters[jj][kk]].crom[i];
                }
                auxc = auxc/clusters[jj].size();
                centroide.push_back(auxc);
            }
            
            //---------------------------------------
            /*
            int aux=0, i_aux;
            for (size_t kk=0;kk<clusters[jj].size();kk++)
                if (inPOB.individuos[clusters[jj][kk]].rango>aux){
                    aux = inPOB.individuos[clusters[jj][kk]].rango;
                    i_aux = kk;
                }
            for (i=0;i<inPOB.lcrom;i++)
                centroide.push_back(inPOB.individuos[clusters[jj][i_aux]].crom[i]); 
            */
            //---------------------------------------
            
            for (size_t kk=0;kk<clusters[jj].size();kk++)
            {            
                inPOB.individuos[clusters[jj][kk]].sFitness = inPOB.individuos[clusters[jj][kk]].sFitness*(1.0 - sharing_fun(distancia(inPOB.individuos[clusters[jj][kk]], centroide, inPOB.lcrom), sigma_share, alfa_share));
            }
            
            centroide.clear();
        }
    }

    vector <short> acumSFitness;
    acumSFitness.resize(mxrango);
    for (k=0;k<mxrango;k++) acumSFitness[k]=1.0;
        
    for (j=0;j<inPOB.tampob;j++) 
        acumSFitness[inPOB.individuos[j].rango-1] += inPOB.individuos[j].sFitness;                
    for (j=0;j<inPOB.tampob;j++) {
        if (acumSFitness[inPOB.individuos[j].rango-1]<=0.0) acumSFitness[inPOB.individuos[j].rango-1] = 1.0;
    }    

    max = 0;
    // recalcular fitness
    for (j=0;j<inPOB.tampob;j++)
    {   
        if (FitnessNRScale)
            inPOB.individuos[j].Fitness = inPOB.individuos[j].Fitness*inPOB.individuos[j].sFitness*inPOB.individuos[j].nr/acumSFitness[inPOB.individuos[j].rango-1];
        else 
            inPOB.individuos[j].Fitness = inPOB.individuos[j].Fitness*inPOB.individuos[j].sFitness/acumSFitness[inPOB.individuos[j].rango-1];
        
        if (max<inPOB.individuos[j].Fitness)  max = inPOB.individuos[j].Fitness;             
    }  
 
    inPOB.Current_Front_Size = 0;
    unsigned aux_rank=inPOB.tampob;
    for (size_t jj=0;jj<inPOB.individuos.size();jj++) 
        if (aux_rank > inPOB.individuos[jj].rango) aux_rank = inPOB.individuos[jj].rango;            
    for (size_t jj=0;jj<inPOB.individuos.size();jj++) 
        if (aux_rank == inPOB.individuos[jj].rango) inPOB.Current_Front_Size++;  
       
    //toc();
    
    return;    
}



// crea las subpoblaciones y las hace evolucionar
//===================================================================
void AG::EvoSubPobs(int brecha, int seltype, int mutatype, float pmuta, int nproc, int Nsubpobs, int tamSubPob, int NGenSubPob, string cfg_settings)
{
    double elapsed;
    short c, i, k, j;
    float auxfit, imax;
    individuo elegido, auxiliar;
    vector <short> elegidos;
    bool desorden, repetidos;
    individuo borrador;
    short eraser;
    vector <short> indice;
    bool sp_c_eval = true;
    
    Dictionary SETTINGS(cfg_settings.c_str());
     
    string T_reemplazo = SETTINGS.get_str("SubPob_Replace_Type");
    sp_c_eval = SETTINGS.get_bool("SP_Classifier_Evaluation");

    for (j=0;j<pobnueva.tampob;j++) indice.push_back(j);    
    // ordenamiento de la población
    k = 0;    
    do {
      desorden = false;
      for (j=0;j<(pobnueva.tampob-(k+1));j++)
      {
        //if (pobnueva.individuos[indice[j]].rango > pobnueva.individuos[indice[j+1]].rango)
        if (pobnueva.individuos[indice[j]].Fitness < pobnueva.individuos[indice[j+1]].Fitness)
        {
          
              eraser = indice[j];
              indice[j] = indice[j+1];
              indice[j+1] = eraser;
              desorden = true;
          
              /* optimizo arriba usando un indice
              eraser = pobnueva.individuos[j];
              pobnueva.individuos[j] = pobnueva.individuos[j+1];
              pobnueva.individuos[j+1] = eraser;
              */
              desorden = true;  
        }
      }
      k++;
    }  while (desorden);
    // guardo los Nsubpobs elegidos
    //--------------------------------//


    if (Nsubpobs>pobnueva.tampob) Nsubpobs = pobnueva.tampob;
    
    int Nf,Ns=0; j=0;
    while ((Ns<Nsubpobs)&&(j<pobnueva.tampob)) {   
        Nf = Verificar(&pobnueva.individuos[indice[j]]);
        if (Nf>=1){
           elegidos.push_back(indice[j]);
           Ns++;
        }   
        j++;
    }    
    if (Ns<Nsubpobs) Nsubpobs=Ns;
    
    // for (j=(Nsubpobs-1);j>=0;j--) elegidos.push_back(indice[j]);

    // para evitar sub-poblaciones repetidas
    do {
      repetidos = false;
      desorden = false;
      
      for (j=0;j<(Nsubpobs-1);j++)  {
        
          for (int jj=j+1;jj<Nsubpobs;jj++)  {
          for (k=0;k<pobnueva.lcrom;k++)
              if (pobnueva.individuos[elegidos[j]].crom[k] != pobnueva.individuos[elegidos[jj]].crom[k]) desorden = true;
        
          if (!desorden) {
            repetidos = true;
            for (k=jj;k<Nsubpobs;k++){
              if (pobnueva.tampob<(elegidos[k]+1)) elegidos[k]+=1; else repetidos = false;
            }  
          } else repetidos = false;
          }
      }
    }  while (repetidos);
   
    
    //          SUBPOBLACIONES          //
    //----------------------------------//
    
    poblacion subpob[Nsubpobs];
    
    bool replace_flag = true;
    
    // Recorro de las subpobs
    for (j=0;j<Nsubpobs;j++) {

        imax = 0;
      
        subpob[j].individuos.resize(tamSubPob);
        subpob[j].tampob = tamSubPob;
      
        /*
        subpob[j].lcrom = 0;              
        for (k=0;k<subpob[j].lcrom;k++) { 
            if (pobnueva.individuos[elegidos[j]].crom[k]) subpob[j].lcrom++;
        } 
        */
        subpob[j].lcrom = Verificar(&pobnueva.individuos[elegidos[j]]);
        
        // Inicializacion de cada subpob
        for (i=0;i<tamSubPob;i++)
        {
            subpob[j].individuos[i].crom.resize(subpob[j].lcrom);
            subpob[j].individuos[i].index.resize(0);
            subpob[j].individuos[i].aptitud.resize(nObjctvs);
            subpob[j].individuos[i].edad = 0;
            // subpob[j].individuos[i].distancias.resize(tamSubPob);
            
            for (k=0;k<pobnueva.lcrom;k++) { 
            // if (!pobnueva.individuos[elegidos[j]].crom[k]) 
               if (pobnueva.individuos[elegidos[j]].crom[k]) subpob[j].individuos[i].index.push_back(k);
            }  
            if (i==0) { // 1ero copio el original con todos los genes
                   for (k=0;k<subpob[j].lcrom;k++) { subpob[j].individuos[i].crom[k] = 1; }
                   subpob[j].individuos[i].aptitud = pobnueva.individuos[elegidos[j]].aptitud;
            } else {      // inicializo los otros aleatoriamente
                   // for (k=0;k<subpob[j].lcrom;k++) { subpob[j].individuos[i].crom[k] = flip(0.5); }           // old init
                   subpob[j].individuos[i].crom = initcrom(subpob[j].lcrom, activ_rate_sp);                      // new init
                   for (int ij=0;ij<(nObjctvs);ij++){ subpob[j].individuos[i].aptitud[ij] = 0; }
            }  
            subpob[j].individuos[i].padre1 = 0;
            subpob[j].individuos[i].padre2 = 0;
        }    
        
       
        auxfit = 0;
        c = 0;
        elegido = subpob[j].individuos[0];
        for (int g=0; g<NGenSubPob; g++){

            if (verbose) { 
                cout << "Generacion " << gen << ", SubPob "<< j+1 << ", Generacion SubPob " << g+1 << endl;            
            }
            tic();
            
            if (c>(NGenSubPob/10)) g = NGenSubPob;
            
            subpob[j] = generSubPob(subpob[j], brecha, seltype, mutatype, pmuta, nproc, imax, g, NGenSubPob, T_reemplazo, sp_c_eval);
            elegido = subpob[j].individuos[0];

            elapsed = toc(verbose);

            if (auxfit < elegido.Fitness)
            {
               auxfit = elegido.Fitness;               
               c = 0;

            } else c = c + 1;
            
        }
        
        if ((T_reemplazo.compare("reemplazo_padre") == 0)||(T_reemplazo.compare("None") == 0)) {
            // reemplazo solo el padre de la subpob si su fitness es mejorado
            
            // Evaluo si tengo que reemplazar
            // if (elegido.aptitud[0] >= pobnueva.individuos[elegidos[j]].aptitud[0]){
            // if (elegido.Fitness >= pobnueva.individuos[elegidos[j]].Fitness){   ->> no es comparable un Fitness de la pob general con uno de subpob
            
            replace_flag = true;
            
            short nObj;
            if (nObjctvs>=2) nObj=2;  
            for (k=0;k<nObj;k++)                
                if (elegido.aptitud[k] < pobnueva.individuos[elegidos[j]].aptitud[k]) replace_flag = false;
            
            if (replace_flag){                
                // Reemplazar individuo
                for (k=0;k<subpob[j].lcrom;k++){
                    if (elegido.crom[k]) 
                       pobnueva.individuos[elegidos[j]].crom[elegido.index[k]] = 1;  // en realidad no seria necesario
                    else
                       pobnueva.individuos[elegidos[j]].crom[elegido.index[k]] = 0;
                    pobnueva.individuos[elegidos[j]].edad = 0;
                }
                
                if (verbose) {
                    cout << "Generacion " << gen << ", SubPob "<< j+1 << " Reemplazo! " << endl;
                    string notify="Reemplazo de individuo de SubPob a POBLACION";
                    Notificar(notify);
                }
                
                for (k=0;k<nObjctvs;k++) pobnueva.individuos[elegidos[j]].aptitud[k]=elegido.aptitud[k];
                // Imprimir(pobnueva.individuos[elegidos[j]], gen, elegidos[j]);
            
            } else {
                if (verbose) {
                    cout << "Generacion " << gen << ", SubPob "<< j+1 << " Mantengo. " << endl;
                }
            } 
        }

        CalcularDistancias(pobnueva, ModifyRepeated);
        CalcularFitness(pobnueva);
    }
    
    if ((T_reemplazo.compare("reemplazo_completo") == 0)||(T_reemplazo.compare("reemplazo_seleccion") == 0)) {
        // reemplazo todos los individuos de la pob general que sean mejorados por individuos de las subpobs
        
        auxiliar.evaluate = true;
        auxiliar.crom.resize(pobnueva.lcrom);
        auxiliar.aptitud.resize(nObjctvs);
        auxiliar.Fitness = 0;
        auxiliar.edad = 0;
        int original_tampob=pobnueva.tampob, icount=pobnueva.tampob;
        //  CREO SUPER-POBLACION
        for (j=0;j<Nsubpobs;j++) {
            
            for (i=0;i<subpob[j].tampob;i++){
            
                for (k=0;k<pobnueva.lcrom;k++) auxiliar.crom[k] = 0;
                
                for (k=0;k<subpob[j].lcrom;k++){
                   if (subpob[j].individuos[i].crom[k]) auxiliar.crom[subpob[j].individuos[i].index[k]] = 1;  
                }
                
                for (int ij=0;ij<(nObjctvs);ij++)
                    auxiliar.aptitud[ij] =  subpob[j].individuos[i].aptitud[ij];
                
                pobnueva.individuos.push_back(auxiliar);
                icount++;
            }
        }
        
        pobnueva.tampob = icount;
        CalcularDistancias(pobnueva, ModifyRepeated);
        CalcularFitness(pobnueva);

        if (T_reemplazo.compare("reemplazo_completo") == 0)
        {
        
            // ordenamiento de la SUPER-POBLACION
            k = 0;    
            do {
            desorden = false;
            for (j=0;j<((short) pobnueva.individuos.size()-(k+1));j++)
            {
                // if (pobnueva.individuos[j].aptitud[0] < pobnueva.individuos[j+1].aptitud[0])           //       <-- podría ser también
                if (pobnueva.individuos[j].Fitness < pobnueva.individuos[j+1].Fitness)                    // parece la mejor opcion
                // if (pobnueva.individuos[j].rango > pobnueva.individuos[j+1].rango)    -> no parece buena opcion segun resultados prelim
                {
                    borrador = pobnueva.individuos[j];
                    pobnueva.individuos[j] = pobnueva.individuos[j+1];
                    pobnueva.individuos[j+1] = borrador;
                    desorden = true;
                }
            }
            k++;
            
            }  while (desorden);
            
            // TRUNCADO DE LA SUPER-POBLACION
            pobnueva.tampob = original_tampob;
            pobnueva.individuos.resize(pobnueva.tampob); 
            
        }
        
        if (T_reemplazo.compare("reemplazo_seleccion") == 0) 
        {
            double sumaptitud = 0;
            for (i=0; i<pobnueva.tampob; i++)
                sumaptitud = sumaptitud + pobnueva.individuos[i].Fitness;
            
            pobvieja.tampob = original_tampob;
            pobvieja.individuos.resize(pobnueva.tampob); 
            for (i=0; i<pobvieja.tampob; i++){
                pobvieja.individuos[i] = pobnueva.individuos[seleccion(pobnueva.tampob, sumaptitud, &pobnueva, seltype)];
            }

            pobnueva.tampob = original_tampob;
            pobnueva.individuos.resize(pobnueva.tampob); 
            
            for (i=0; i<pobnueva.tampob; i++){
                pobnueva.individuos[i] = pobvieja.individuos[i];
            }
            
        }
        
        CalcularDistancias(pobnueva, ModifyRepeated);
        CalcularFitness(pobnueva);
        
    }
        
    (void) elapsed;
    //----------------------------------//

    return;
    
}
//===================================================================


// realiza una generacion de una subpoblacion
//===================================================================
poblacion AG::generSubPob(poblacion &SubPob, int brecha, int seltype, int mutatype, float pmutaSP, int nproc, float imax, short subgen, short maxsubgen, string T_reemplazo, bool sp_c_eval)
{
    int i, k, j, mate1, mate2;
    unsigned aux;
    int punto, bc = 1;
    poblacion subpobnueva;
    double sumaptitud = 0;
    
    // Inicializacion subpob auxiliar
    subpobnueva.individuos.resize(SubPob.tampob);
    subpobnueva.lcrom = SubPob.lcrom;
    subpobnueva.tampob = SubPob.tampob;
    SubPob.NMutas = 0;
    for (i=0;i<SubPob.tampob;i++)
    {
        subpobnueva.individuos[i].crom.resize(subpobnueva.lcrom);
        subpobnueva.individuos[i].aptitud.resize(nObjctvs);
        subpobnueva.individuos[i].index.resize(subpobnueva.lcrom);
        /*
        subpobnueva.individuos[i].index.resize(0);
        for (j=0;j<subpobnueva.lcrom;j++)      
            subpobnueva.individuos[i].index.push_back(SubPob.individuos[i].index[j]);
        */
    }
    
    //Busco el mejor y lo copio en la nueva generación (elitismo)
    // acum = SubPob.individuos[0].Fitness;
    aux = SubPob.individuos[0].rango;
    punto = 0;
    for (j=1;j<SubPob.tampob;j++)
    {
        sumaptitud = sumaptitud + SubPob.individuos[j].Fitness;
        /*
        if (SubPob.individuos[j].Fitness > acum)
        {
              acum = SubPob.individuos[j].Fitness;
              punto = j;
        }
        */
        if (SubPob.individuos[j].rango < aux)
        {
              aux = SubPob.individuos[j].rango;
              punto = j;
        }
    }
 
    //aca tengo j con el mayor fitness
    subpobnueva.individuos[0] = SubPob.individuos[punto];
    
    if (subgen==0) // primera sub-generacion
    {
      for (i=0; i<SubPob.tampob; i++)
      subpobnueva.individuos[i] = SubPob.individuos[i];
    } else {
      // brecha generacional
      for (i=1; i<=brecha; i++)
      subpobnueva.individuos[i] = SubPob.individuos[seleccion(SubPob.tampob, sumaptitud, &SubPob, seltype)];
    }

    //--------------------------------//
  
    cromosoma newbie1, newbie2;

    if (subgen!=0){
      j = brecha+1;
      do
      {
      mate1 = seleccion(SubPob.tampob, sumaptitud, &SubPob, seltype);
      mate2 = seleccion(SubPob.tampob, sumaptitud, &SubPob, seltype);

      if (bc < brecha)
      {
          subpobnueva.individuos[bc] = SubPob.individuos[mate1];
          bc++;
      }

      newbie1 = mutacion(SubPob.individuos[mate1].crom, pmutaSP, mutatype, SubPob.NMutas);
      newbie2 = mutacion(SubPob.individuos[mate2].crom, pmutaSP, mutatype, SubPob.NMutas);

     if (j<(SubPob.tampob-1)) cruza(newbie1, newbie2, &subpobnueva.individuos[j].crom, &subpobnueva.individuos[j+1].crom, subpobnueva.lcrom,  pcruza);
     else  cruza(newbie1, newbie2, &subpobnueva.individuos[j].crom, &subpobnueva.individuos[j].crom, subpobnueva.lcrom,  pcruza);

      subpobnueva.individuos[j].padre1 = mate1;
      subpobnueva.individuos[j].padre2 = mate2;

      if (j<(SubPob.tampob-1))
      {
          subpobnueva.individuos[j+1].padre1 = mate2;
          subpobnueva.individuos[j+1].padre2 = mate1;
      }

      j = j + 2;

      } while ( j <= SubPob.tampob-1);
    }
    
    CalcularDistancias(subpobnueva, ModifyRepeated);
 
    /*------ MPI VARIOS + Var Paralelizacion ------*/

    MPI_Request request[nproc];
    MPI_Status status;
    int buffer[pobnueva.lcrom];
    int flag;
    bool busys[nproc];
    double aptitud[nproc][nObjctvs];
    int map[nproc];
    int tag;
    int params[3];

    /*---------------------------------------------*/

    params[0] = pobnueva.lcrom;  
    params[1] = 1;       // 0 indica pob principal; 1 indica subpob
    params[2] = 0;       // indica si se debe usar el clasificador para evaluar
    if (sp_c_eval) params[2] = 1;
    
    if (T_reemplazo.compare("reemplazo_completo") == 0)
        if (subgen >= (maxsubgen-1))
            params[2] = 1;     // con reemplazo completo en la ultima generacion evaluo a todos con clasificador 
    
    // repartir cálculos de aptitud entre nodos
    for (i=0;i<nproc;i++) busys[i] = false;
    tag = 1000 + nproc;

    if (subgen==0) j = 1; else j = brecha+1;
    
    while (j<SubPob.tampob)
    {
       Verificar(&subpobnueva.individuos[j]);
       for (i=0;((i<nproc)&&(j<SubPob.tampob));i++)
       {
           if  (!busys[i])
           {
               MPI_Send(params, 3, MPI_INTEGER, i, tag, everyone);        
               /*
               MPI_Send(&seed, 1, MPI_FLOAT, i, tag, everyone);  
               MPI_Send(&nObjctvs, 1, MPI_INTEGER, i, tag, everyone);
               */
       
               map[i]=j;
               for (k=0;k<pobnueva.lcrom;k++) buffer[k] = 0;
               for (k=0;k<subpobnueva.lcrom;k++)
               {                  
                  if (subpobnueva.individuos[j].crom[k]) buffer[subpobnueva.individuos[j].index[k]] = 1;
               }  
               
               // cout << ">> sp " << j << " << lcrom " << lcrom <<  " subpobnueva.lcrom " << subpobnueva.lcrom << " last-index " << subpobnueva.individuos[j].index[subpobnueva.lcrom-1];
               // for (k=0;k<lcrom;k++) cout << buffer[k] << "  ";
               // cout << "\n";
               
               MPI_Send(buffer, pobnueva.lcrom, MPI_INTEGER, i, tag, everyone);
               busys[i] = true;
               MPI_Irecv(&aptitud[i], nObjctvs, MPI_DOUBLE, i, tag, everyone, &request[i]);
               j++;
           }
       }
    
       for (i=0;i<nproc;i++)
       {
           flag = 0;
           if (busys[i]) MPI_Test(&request[i],&flag,&status);
           if (flag == 1)
           {
               busys[i] = false;
               // subpobnueva.individuos[map[i]].aptitud = aptitud[i][0];
               for (int ij=0;ij<nObjctvs;ij++)
               {    
                   subpobnueva.individuos[map[i]].aptitud[ij] = aptitud[i][ij];
                   if (aptitud[i][ij]>aptitud_max[ij]) aptitud_max[ij]=aptitud[i][ij];
                   if (aptitud[i][ij]<aptitud_min[ij]) aptitud_min[ij]=aptitud[i][ij];               
               }    
               if (imax<aptitud[i][0])
               {
                   imax = aptitud[i][0];
                   // Imprimir(subpobnueva.individuos[map[i]], gen, map[i]);
               }
           }
       }

    }
   
    for (i=0;i<nproc;i++)
    {
       if (busys[i])
       {
           flag = 0;
           while (flag!=1) MPI_Test(&request[i],&flag,&status);
           busys[i] = false;
           // subpobnueva.individuos[map[i]].aptitud = aptitud[i][0];
           for (int ij=0;ij<nObjctvs;ij++) 
           {
               subpobnueva.individuos[map[i]].aptitud[ij] = aptitud[i][ij];               
               if (aptitud[i][ij]>aptitud_max[ij]) aptitud_max[ij]=aptitud[i][ij];
               if (aptitud[i][ij]<aptitud_min[ij]) aptitud_min[ij]=aptitud[i][ij];               
           }

           
           int kk=0;
           for (int rr=0;rr<subpobnueva.lcrom;rr++) if (subpobnueva.individuos[map[i]].crom[rr]) kk++;
           
           // cout << "........................................................> " << subpobnueva.individuos[map[i]].aptitud[1] << " : " << kk << endl;
           
           if (imax<aptitud[i][0])
           {
              imax = aptitud[i][0];
              // Imprimir(subpobnueva.individuos[map[i]], gen, map[i]);
           }
       }
    }
  
    CalcularFitness(subpobnueva);    

    // ordenamiento de la sub-población    
    individuo eraser;
    bool desorden;
    k = 0;    
    do {
      desorden = false;
      for (j=0;j<(SubPob.tampob-(k+1));j++)
      {
        // if (subpobnueva.individuos[j].aptitud[0] < subpobnueva.individuos[j+1].aptitud[0])
        if (subpobnueva.individuos[j].Fitness < subpobnueva.individuos[j+1].Fitness)
        {
              eraser = subpobnueva.individuos[j];
              subpobnueva.individuos[j] = subpobnueva.individuos[j+1];
              subpobnueva.individuos[j+1] = eraser;
              desorden = true;
        }
      }
      k++;
    }  while (desorden);

    
    // ----------------------------
    
    if ((!sp_c_eval)&&((T_reemplazo.compare("reemplazo_padre") == 0)||(T_reemplazo.compare("None") == 0))&&(subgen >= (maxsubgen-1))) 
    {  
       // con reemplazo_padre evaluo el mejor indidividuo (j=0) con el clasificador en la ultima generacion       
       params[2] = 1; 
       j = 0; 
       Verificar(&subpobnueva.individuos[j]);

       MPI_Send(params, 3, MPI_INTEGER, 0, tag, everyone);        
       for (k=0;k<pobnueva.lcrom;k++) buffer[k] = 0;
       for (k=0;k<subpobnueva.lcrom;k++)
       {                  
           if (subpobnueva.individuos[j].crom[k]) buffer[subpobnueva.individuos[j].index[k]] = 1;
       }  
       MPI_Send(buffer, pobnueva.lcrom, MPI_INTEGER, 0, tag, everyone);
       MPI_Irecv(&aptitud[0], nObjctvs, MPI_DOUBLE, 0, tag, everyone, &request[0]);
       flag = 0;
       while (flag!=1) MPI_Test(&request[0],&flag,&status);
       if (flag == 1)
       {
           for (int ij=0;ij<nObjctvs;ij++)
           {    
               subpobnueva.individuos[j].aptitud[ij] = aptitud[0][ij];
           }    
       }
    }

    // ----------------------------
    
    SubPob = subpobnueva;
    
    return(SubPob);
  
}
//===================================================================



double AG::sharing_fun(double dist, double sigma, double alfa)
{
    double sharing = 0.0;
    
    if (dist < sigma)
        sharing = ( 1.0 - pow((dist/sigma),alfa) );
    // cout << " --> " << dist << endl;
    return sharing;
}





//===================================================================
void AG::inicializar(int in_tampob, int in_lcrom, int in_maxgen, double in_pcruza, double in_pmutacion, int nproc, float tasa_activ, string cfg_settings)
{
     int i, j, k;

     //Inicializa datos.
     pobvieja.tampob = in_tampob;
     pobnueva.tampob = in_tampob;
     pobvieja.lcrom = in_lcrom;
     pobvieja.NMutas = 0;
     pobnueva.lcrom = in_lcrom;
     maxgen = in_maxgen;
     pcruza = in_pcruza;
     pmutacion = in_pmutacion;
     max = 0;
     activ_rate = tasa_activ;

     Dictionary SETTINGS(cfg_settings.c_str());
     
     string filename;     
     
     if (nsubpob>0) 
     {
          filename  = folder+"/"+"MOELIGA+Subpobs_results_"+fecha+ string_aux_filename +".txt";
          crom_file = folder+"/"+"MOELIGA+Subpobs_results_"+fecha+ string_aux_filename +".crom";
          yaml_file = folder+"/"+"MOELIGA+Subpobs_results_"+fecha+ string_aux_filename +".train";
     } else 
     {
          filename  = folder+"/"+"MOELIGA_results_"+fecha+ string_aux_filename +".txt";
          crom_file = folder+"/"+"MOELIGA_results_"+fecha+ string_aux_filename +".crom";
          yaml_file = folder+"/"+"MOELIGA_results_"+fecha+ string_aux_filename +".train";
     }
     
     res_file = filename;
     /*
     filename.insert(0," > ");
     filename.insert(0,cfg_settings.c_str());     
     filename.insert(0,"cat ");
     system(filename.c_str());
     */

     aptitud_min.resize(nObjctvs);
     aptitud_max.resize(nObjctvs);
     for (i=0;i<nObjctvs;i++){
         aptitud_min[i]=__DBL_MAX__;
         aptitud_max[i]=__DBL_MIN__;
     }
     // Inicializacin de los vectores:
     pobnueva.individuos.resize(pobnueva.tampob);
     pobvieja.individuos.resize(pobnueva.tampob);
     // pobnueva.orden.resize(tampob);
     // pobvieja.orden.resize(tampob);

     for (i=0;i<pobvieja.tampob;i++)
     {
         pobnueva.individuos[i].evaluate = true;
         pobvieja.individuos[i].evaluate = true;
         pobnueva.individuos[i].crom.resize(pobvieja.lcrom);     
         pobvieja.individuos[i].crom.resize(pobvieja.lcrom);
         pobvieja.individuos[i].aptitud.resize(nObjctvs);
         pobnueva.individuos[i].aptitud.resize(nObjctvs);
         pobnueva.individuos[i].edad = 0;
         // pobnueva.individuos[i].distancias.resize(pobvieja.tampob);
         // pobvieja.individuos[i].distancias.resize(pobvieja.tampob);
     }

     /*------ MPI VARIOS + Var Paralelizacion ------*/

     MPI_Request request[nproc];
     MPI_Status status;
     int buffer[pobvieja.lcrom];
     int flag;
     bool busys[nproc];
     double aptitud[nproc][nObjctvs];
     int map[nproc];
     double *pointApt;
     int tag;
     char *arg[3];
     int params[3];

     /*--------------------------------------------*/

     // Iniciar los procesos en todos los nodos
     for (i=0;i<nproc;i++)
     {
          busys[i] = false;
     }

     tag = 1000 + nproc;
     arg[0] = itoa(tag,10);
     arg[1] =  (char*) cfg_settings.c_str();
     arg[2] = NULL;     

     MPI_Comm_spawn((char*) slvbin.c_str(), arg, nproc, MPI_INFO_NULL, 0, MPI_COMM_SELF, &everyone, MPI_ERRCODES_IGNORE);

     params[0] = pobvieja.lcrom;  
     params[1] = 0;  // indica tipo de poblacion
     params[2] = 1;  // indica si debe usar clasificador para evaluar

     for (i=0;i<nproc;i++)
     {
         // MPI_Send(params, 2, MPI_INTEGER, i, tag, everyone);
         MPI_Send(&alfa, 1, MPI_FLOAT, i, tag, everyone);  
         MPI_Send(&nObjctvs, 1, MPI_INTEGER, i, tag, everyone);
     }
      
     // Inicializar poblacion
     // y repartir calculos de aptitud entre nodos
     j = 0;
     while (j<pobvieja.tampob)
     {
          
          for (i=0;((i<nproc)&&(j<pobvieja.tampob));i++)
          {              
             if (stepped_activ) { 
                if (j<=0.55*pobvieja.tampob)
                    pobvieja.individuos[j].crom = initcrom(pobvieja.lcrom, activ_rates[0]);                                     
                if ((j>0.55*pobvieja.tampob)&&(j<=0.85*pobvieja.tampob))
                    pobvieja.individuos[j].crom = initcrom(pobvieja.lcrom, activ_rates[1]);
                if (j>0.85*pobvieja.tampob)
                    pobvieja.individuos[j].crom = initcrom(pobvieja.lcrom, activ_rates[2]);
             } else 
             {
                pobvieja.individuos[j].crom = initcrom(pobvieja.lcrom, activ_rate);                                     // new init
             }
             Verificar(&pobvieja.individuos[j]);
                           
             pobvieja.individuos[j].padre1 = 0;
             pobvieja.individuos[j].padre2 = 0;

            
             if  (!busys[i])
             {
                 MPI_Send(params, 3, MPI_INTEGER, i, tag, everyone);
                 map[i]=j;
                 for (k=0;k<pobvieja.lcrom;k++)
                    if (pobvieja.individuos[j].crom[k]) buffer[k] = 1; else buffer[k] = 0;
                 MPI_Send(buffer, pobvieja.lcrom, MPI_INTEGER, i, tag, everyone);
                 // MPI_Send(&seed, 1, MPI_INTEGER, i, tag, everyone);  // envio semilla
                 busys[i] = true;
                 pointApt=&(aptitud[i][0]);
                 MPI_Irecv(pointApt, nObjctvs, MPI_DOUBLE, i, tag, everyone, &request[i]);
                 j++;
             }
          }
          
          for (i=0;i<nproc;i++)
          {
              flag = 0;
              if (busys[i]) MPI_Test(&request[i],&flag,&status);
              if (flag == 1)
              {
                 busys[i] = false;
                 for (k=0;k<nObjctvs;k++) {
                     pobvieja.individuos[map[i]].aptitud[k] = aptitud[i][k];  // MOGA
                     if (aptitud[i][k]>aptitud_max[k]) aptitud_max[k]=aptitud[i][k];
                     if (aptitud[i][k]<aptitud_min[k]) aptitud_min[k]=aptitud[i][k];
                 }    
              }
          }
    }

    for (i=0;i<nproc;i++)
    {
        if (busys[i])
        {
           flag = 0;
           while (flag!=1) MPI_Test(&request[i],&flag,&status);
           busys[i] = false;
           // for (k=0;k<nObjctvs;k++) pobvieja.individuos[map[i]].aptitud[k] = aptitud[i][k];  // MOGA
           for (k=0;k<nObjctvs;k++) {
               pobvieja.individuos[map[i]].aptitud[k] = aptitud[i][k];  // MOGA
               if (aptitud[i][k]>aptitud_max[k]) aptitud_max[k]=aptitud[i][k];
               if (aptitud[i][k]<aptitud_min[k]) aptitud_min[k]=aptitud[i][k];
           }    
        }
    }
    
    
    CalcularDistancias(pobvieja, false); // son individuos "recien creados" -> no es necesario aplicar operador_diversidad
    CalcularFitness(pobvieja);
    

    return;

}
//===================================================================




//===================================================================
int AG::seleccion(int tampob, float sumaptitud, poblacion *pob, int caso)
{
  double random, sumaparc; // puntero de la ruleta, suma parcial
  double frandom;
  int j, k=0, irandom;

  int tourn_size = ((int) (((double) tampob)*15.0/100));
 
  poblacion &inpob = *pob;

  switch ( caso )
  {
     case 1 :   // ruleta
     {
             j = 0;
             sumaptitud = 0;
             do {
                   sumaptitud = sumaptitud + inpob.individuos[j].Fitness;
                   j = j + 1;
                   
             } while (j < tampob-1);
      
             sumaparc = 0;

             irandom = (rand()%100);
             frandom = irandom;
             frandom = frandom / 100;
             random = frandom * sumaptitud;

             j = 0;
             do {
                   sumaparc = sumaparc + inpob.individuos[j].Fitness;
                   j = j + 1;
             } while ((sumaparc < random) & (j < tampob-1));

             return j;

     }   // fin case 1

     case 2 : // torneo
     {
         if (tourn_size>tampob) tourn_size = tampob;
         
         vector < int > chosen(tourn_size);

         for  (j=0;j<tourn_size;j++)
         {
              irandom = (rand()%tampob);
              chosen[j] = irandom;
         }
         double maxfit = inpob.individuos[chosen[0]].Fitness;
         k = chosen[0];
         for  (j=1;j<tourn_size;j++)
         {
              if (inpob.individuos[chosen[j]].Fitness > maxfit)
              {
                  k = chosen[j];
                  maxfit = inpob.individuos[k].Fitness;
              }
         }
         return k;
     }   // fin case 2

     case 3 :   // ventanas
     {
             individuo eraser;
             bool desorden;
             k = 0;
             // ordenamiento de la poblacion
             do {
                    desorden = false;
                    for (j=0;j<(tampob-(k+1));j++)
                    {
                          if (inpob.individuos[j].Fitness < inpob.individuos[j+1].Fitness)
                          {
                               eraser = inpob.individuos[j];
                               inpob.individuos[j] = inpob.individuos[j+1];
                               inpob.individuos[j+1] = eraser;
                               desorden = true;
                          }
                    }
                    k++;
             }  while (desorden);

             int winsize = tampob*40/100; // max ventana = 40% tamanio de poblacion
             irandom = (rand()%(winsize-1))+1; // tamanio de ventana aleatorio
             irandom = (rand()%irandom); // indice dentro de la ventana

             return irandom;

     }   // fin case 3

  } // fin switch

  return 0;
} // fin seleccion
//===================================================================





//===================================================================
bool AG::flip(float prob)
{
    long int irandom = (rand()%2147483640)+1;
    double frandom;
    frandom = irandom;
    frandom = frandom / 2147483640;
// cout << "-- > flip " << frandom << endl;
    if (prob == 1)
       { return true; }
    else
       { return (frandom <= prob);}
}
//===================================================================




//===================================================================
cromosoma AG::mutacion(cromosoma crom, double pmutacion, int caso, int &NMutas)
{
  cromosoma aux_crom;
  
  
  
  switch ( caso )
  {
      case 1 :   // de cromosoma
        {
                int rand_gen;
                // rand_gen = ( rand() % crom.size())+1;
                rand_gen = (rand() % crom.size());

                aux_crom = crom;
                if (flip(pmutacion)){
                   aux_crom[rand_gen] = !aux_crom[rand_gen];
                   NMutas = NMutas + 1;
                }
          break;
      }

      case 2 :  // de gen
        {
            double pmuta = pmutacion/crom.size();  
            
            unsigned j;
            double pmutaD = pmuta;
            
            aux_crom = crom;
            for (j=0;j<crom.size();j++)
            {
                    // discrimimar segun valor del gen
                    // if (aux_crom[j]) pmutaD = 1.0*pmuta; else pmutaD = 0.7*pmuta;
                      
                    if (flip(pmutaD)){    
                       aux_crom[j] = !aux_crom[j];
                       NMutas = NMutas + 1;
                    } 
            }
            
            break;

      }

  } // fin switch
  
  // unsigned nf = 
  Verificar(&aux_crom);
  
  return aux_crom;
} // fin mutacion
//===================================================================




//===================================================================
void AG::cruza(cromosoma padre1, cromosoma padre2, cromosoma *hijo1, cromosoma *hijo2, int Nlcrom,  float pcruza)
{

   int jcruza, j;

   if (flip(pcruza))
   {
      
      jcruza = (rand()%Nlcrom); 
   }
   else { jcruza = Nlcrom; }

   cromosoma &aux1 = *hijo1;
   cromosoma &aux2 = *hijo2;

   for (j=0;j<jcruza;j++)
   {
      aux1[j] = padre1[j];
      aux2[j] = padre2[j];
   }

   for (j=jcruza;j<Nlcrom;j++)
   {
       aux1[j] = padre2[j];
       aux2[j] = padre1[j];
   }

   j = Verificar(&aux1);
   j = Verificar(&aux2);
   
   return;
}
//===================================================================






//===================================================================
void AG::Notificar(string notify)
{
    // Al archivo resultag.txt
    string filename = res_file;
    if (!results.is_open()) results.open(filename.c_str(), ofstream::out | ofstream::app);
    
    results << endl;
    results << ":: --> < AVISO >: " << notify.c_str() << endl;
    results << endl;

    results.close();
}
//===================================================================


//===================================================================

unsigned AG::Verificar(individuo *JohnDoe)
{

    individuo &indiv = *JohnDoe;
    unsigned nf = 0;
    
    for (unsigned j=0;j<indiv.crom.size(); j++){
        if (indiv.crom[j]) nf++;
    }
    indiv.nF = nf;
    
    if (nf==0)
    {
       int rand_gen;
       rand_gen = (rand() % indiv.crom.size());
       indiv.crom[rand_gen] = true; 
       nf++;
       indiv.nF = nf;
       indiv.edad = 0;
    }
    
    return nf;
}


unsigned AG::Verificar(cromosoma *JohnDoe)
{

    cromosoma &cromo = *JohnDoe;
    unsigned nf = 0;
    
    for (unsigned j=0;j<cromo.size(); j++){
        if (cromo[j]) nf++;
    }
    
    if (nf==0)
    {
       int rand_gen;
       rand_gen = (rand() % cromo.size());
       cromo[rand_gen] = true; 
       nf++;
    }
    
    return nf;
}


void AG::ImprimirCromoForTest(individuo johndoe, bool append)
{
    // guardo los cromosoma en archivo separado para correr los tests
    
    ofstream best_crom;
    if (!append) best_crom.open(crom_file.c_str(), ofstream::out | ofstream::trunc);
    else best_crom.open(crom_file.c_str(), ofstream::out | ofstream::app); 
    
    for (size_t i=0;i<johndoe.crom.size();i++) if (johndoe.crom[i]) best_crom << i+1 << " ";
    best_crom << endl;     
    
    best_crom.close();     
    
}


void AG::ImprimirFrente(poblacion &inPOB, int generac, double best_fit, bool onlyBest, bool yaml)
{
    // unsigned best_j = 0;
    short cnt = 0;

    unsigned aux_rank=inPOB.tampob;
    for (unsigned j=0;j<inPOB.individuos.size();j++) 
    {
        if (aux_rank > inPOB.individuos[j].rango){
            aux_rank = inPOB.individuos[j].rango;            
        }    
    } 
    
    for (unsigned j=0;j<inPOB.individuos.size();j++)
    { 
            if (aux_rank == inPOB.individuos[j].rango)
            {        
                    if (yaml) 
                    {
                        if (cnt>0) yaml_ImprimirCromo(inPOB.individuos[j], generac, j, false);
                        else yaml_ImprimirCromo(inPOB.individuos[j], generac, j, true);
                    } 
                    else 
                    {                       
                        if (cnt>0) ImprimirCromo(inPOB.individuos[j], generac, j, false);
                        else ImprimirCromo(inPOB.individuos[j], generac, j, true);
                    }
                    
                    if (!onlyBest) 
                    {
                        if (cnt>0) ImprimirCromoForTest(inPOB.individuos[j], true);
                        else ImprimirCromoForTest(inPOB.individuos[j], false);
                    }
                    cnt++;
            }  
    }    
    
    if (onlyBest) {
         int Best = 0;
         double dist = 0.0, aux=0.0;
         
         for (unsigned j=0;j<inPOB.individuos.size();j++)
         { 
            if (aux_rank == inPOB.individuos[j].rango)  
            {
                 if (Rfun==1) 
                 {    
                     aux = inPOB.individuos[j].R1;
                     
                 } else { // if (Rfun==2) // por omision
                         
                     aux = inPOB.individuos[j].R2;
                 }                       
                 if (aux>dist) {  // maximizar R1 o R2
                     dist = aux;
                     Best = j;
                 }                    
            }  
        }

        ImprimirCromoForTest(inPOB.individuos[Best], false);
    
    }    
    
}

//===================================================================







//===================================================================
void AG::ImprimirCromo(individuo johndoe, int generac, int indiv, bool newfront)
{
     indiv++; 
     
     if (!results.is_open()) results.open(res_file.c_str(), ofstream::out | ofstream::app);
     
     results << endl;
     results << "::> Generacion: " << generac << endl;
     results << "::> Individuo: " << indiv << endl;
     results << "::> Coeficientes Seleccionados: " <<  endl;
     for (size_t i=0;i<johndoe.crom.size();i++) if (johndoe.crom[i]) results << i+1 << " ";        
     results << endl;
     results << "::> Numero Coeficientes Seleccionados: " << johndoe.nF << endl;
     results << "::> Fitness: " << johndoe.Fitness << endl;
     results << "::> Shared Fitness: " << johndoe.sFitness << endl;
     results << "::> Rank: " << johndoe.rango << endl;
     for (int i=0;i<(nObjctvs);i++)
         results << "::> Objetivo " << i <<": "  << johndoe.aptitud[i] << endl;

     results << "::> Medida para elegir el mejor R1: " << johndoe.R1 << endl;
     results << "::> Medida para elegir el mejor R2: " << johndoe.R2 << endl;
     
     results.close();
     
     return;
}

//===================================================================





//===================================================================
void AG::yaml_ImprimirCromo(individuo johndoe, int generac, int indiv, bool newfront)
{   
     indiv++;

     if (!results.is_open()) results.open(yaml_file.c_str(), ofstream::out | ofstream::app);

     if (newfront) 
     {    
         results << "  #===================================" << endl;
         results << "  FRENTE_DE_PARETO:" << endl << endl;
     }        
     results << "  #---------------" << endl;
     results << "   - INDIVIDUO: "  << indiv << endl;
     results << "     GENERACION: " << generac << endl;
     results << "     NUMERO_COEFICIENTES_SELECCIONADOS: " << johndoe.nF << endl;
     /*
     results << "     COEFICIENTES_SELECCIONADOS: [";
     bool fflag = false;
     for (int i=0;i<johndoe.crom.size();i++) 
         if (johndoe.crom[i]) {                
             if (fflag) results << ", ";                
             results << i+1;
             fflag = true;                                 
         }               
     results << "]" << endl;
     */
     results << "     FITNESS: " << johndoe.Fitness << endl;
     results << "     SHARED_FITNESS: " << johndoe.sFitness << endl;
     results << "     RANK: " << johndoe.rango << endl;        
     results << "     RANK_COUNT: " << johndoe.nr << endl;
     results << "     AGE: " << johndoe.edad << endl;
     results << "     MEAN_DISTANCE: " << johndoe.mean_dist << endl;
     results << "     NICHE_COUNT: " << johndoe.ncount << endl;        
     results << "     MEDIDA_PARA_ELEGIR_EL_MEJOR_R1: " << johndoe.R1 << endl;
     results << "     MEDIDA_PARA_ELEGIR_EL_MEJOR_R2: " << johndoe.R2 << endl;
     results << "     OBJETIVOS: [";        
     for (int i=0;i<nObjctvs;i++) {
         results << johndoe.aptitud[i];
         if (i<(nObjctvs-1)) results << ", "; else results << "]" << endl << endl;
     }            
     results.close();
     
     return;
}

//===================================================================




//===================================================================

void AG::ImprimirGen(int generac, double maxfitness, double minfitness, double prom, poblacion &inPOB)
{
      time_t rawtime;
      struct tm * timeinfo;

      time ( &rawtime );
      timeinfo = localtime ( &rawtime );

      if (!results.is_open()) results.open(res_file.c_str(), ofstream::out | ofstream::app);
      
      double caux=0.0;
      for (size_t i=0;i<clusters.size();i++) caux=caux+clusters[i].size();
      caux=caux/clusters.size();         
          
      results << " " << endl;
      results << "|--------------------------------------------|" << endl;
      results << "|    Generacion: " << generac << endl;
      results << "|    Peor Fitness: " << minfitness << endl;
      results << "|    Mejor Fitness: " << maxfitness << endl;
      results << "|    Promedio Fitness: " << prom << endl;      
      results << "|    Cantidad de Mutaciones: " << inPOB.NMutas << endl;
      results << "|    Distancia de Media: " << inPOB.mean_dist << endl;      
      results << "|    Cantidad de clusters: " << clusters.size() << endl;      
      results << "|    Promedio cluster: " << (int) caux << endl;
      results << "|    "<< asctime (timeinfo); // << endl;
      results << "|--------------------------------------------|" << endl;
      results << " " << endl;
      results.close();
      
      
}      

//===================================================================



//===================================================================

void AG::yaml_ImprimirGen(int gener, double maxfitness, double minfitness, double prom, poblacion &inPOB)
{
      /*
      time_t rawtime;
      struct tm * timeinfo;
      time ( &rawtime );
      timeinfo = localtime ( &rawtime );
      */
      
      if (!results.is_open()) results.open(yaml_file.c_str(), ofstream::out | ofstream::app);
      
      double caux=0.0;
      for (size_t i=0;i<clusters.size();i++) caux=caux+clusters[i].size();
      if (clusters.size()>0) caux=caux/clusters.size();                   
      
      results << "#####################################"  << endl;
      results << "- GENERATION: " << gener << endl << endl;
      results << "  #===================================" << endl;
      results << "  GENERAL:" << endl << endl;
      results << "   NUMERO_COEFICIENTES_SELECCIONADOS: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].nF;
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      }    
      results << "   FITNESS: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].Fitness;
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      } 
      results << "   SHARED_FITNESS: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].sFitness;
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      } 
      results << "   DISTANCIAS_MEDIAS: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].mean_dist;
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      } 
      results << "   RANK: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].rango;
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      } 
      results << "   NICHE_COUNT: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].ncount;
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      }       
      results << "   OBJETIVO_0: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].aptitud[0];
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      }       
      if (nObjctvs>1) {
        results << "   OBJETIVO_1: [";
        for (short i=0;i<inPOB.tampob;i++) {      
            results << inPOB.individuos[i].aptitud[1];
            if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
        }       
      }
      if (nObjctvs>2) {
        results << "   OBJETIVO_2: [";
        for (short i=0;i<inPOB.tampob;i++) {      
            results << inPOB.individuos[i].aptitud[2];
            if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
        }       
      }      
      results << "   MINIMOS_POR_OBJETIVO: [";
      for (short i=0;i<nObjctvs;i++) {
          results << aptitud_min[i]; 
          if (i<(nObjctvs-1)) results << ", "; else results << "]" << endl;
      }           
      results << "   MAXIMOS_POR_OBJETIVO: [";
      for (short i=0;i<nObjctvs;i++) {
          results << aptitud_max[i]; 
          if (i<(nObjctvs-1)) results << ", "; else results << "]" << endl;
      }     
      
      results << endl;
      /*
      results << "   MEDIDA_PARA_ELEGIR_EL_MEJOR_R1: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].R1;
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      }        
      results << "   MEDIDA_PARA_ELEGIR_EL_MEJOR_R2: [";
      for (short i=0;i<inPOB.tampob;i++) {      
           results << inPOB.individuos[i].R2;
           if (i<(inPOB.tampob-1)) results << ", "; else results << "]" << endl;
      } 
      */
      results << "   NUMERO_DE_VECES_QUE_SE_ELIGE_CADA_FEATURE: [";
      for (short i=0;i<inPOB.lcrom;i++) {      
           results << inPOB.histograma[i];
           if (i<(inPOB.lcrom-1)) results << ", "; else results << "]" << endl;
      }        
      results << endl;
      results << "   CANTIDAD_DE_MUTACIONES: " << inPOB.NMutas << endl;
      // results << "   DISTANCIA_MEDIA_POBLACIONAL: " << inPOB.mean_dist << endl; // redundante, ya esta el vector
      results << "   CANTIDAD_DE_CLUSTERS: "  << clusters.size() << endl;
      results << "   PROMEDIO_CLUSTER: "  << caux << endl;
      results << endl;
      
      double total_elapsed = global_toc(verbose);      
      results << "   TOTAL_ELAPSED_TIME: " << total_elapsed << endl;
      results << endl;
      
      results.close();
      
}      

//===================================================================













//===================================================================
void AG::Terminar(int nproc, int lcrom, bool txt)
{
      int buffer[lcrom];
      int params[3];
      int tag = 1000+nproc;
      
      buffer[0]=-5; // señal para matar todos los procesos creados
      params[0]=lcrom;
      params[1]=0;
      params[2]=1;
      
      for (int i=0;i<nproc;i++)
      { 
          MPI_Send(params, 3, MPI_INTEGER, i, tag, everyone);
          // MPI_Send(&seed, 1, MPI_FLOAT, i, tag, everyone);  
          // MPI_Send(&nObjctvs, 1, MPI_INTEGER, i, tag, everyone);
          MPI_Send(buffer, lcrom, MPI_INTEGER, i, tag, everyone);
      }
      
      if (txt) {
        double total_elapsed = global_toc(verbose);      
        string filename = res_file;
        if (!results.is_open()) results.open(filename.c_str(), ofstream::out | ofstream::app);      
        results << " " << endl;
        results << "TOTAL Time elapsed: " << total_elapsed << endl;
        results << " " << endl;      
        if (results.is_open()) results.close();
      }
      
      MPI_Finalize();
      
}
//===================================================================








//#############################################
// ------------- MAIN FUNCTION --------------
//#############################################
int main(int argc, char** argv)
{
    // string cmd = "mpdallexit; mpdboot; export PATH=$PATH:$PWD";
    string cmd = "mpdboot; export PATH=$PATH:$PWD";
    int bar = system(cmd.c_str());
    
    string aux, cfg_settings = "SETTINGS.cfg";
    
    //=========================================
    // BUSCO PARAMETROS EN LA LINEA DE COMANDOS
    //-----------------------------------------
        
    for (int i=1;i<argc;i++)
    {
        aux = argv[i];

        if (aux == "help")
        {
            cout << " - Algoritmo genético (versión en paralelo) -"<< endl;
            cout << "ag [parametro1 valor1] [parametro2 valor2] [...] "<< endl;
            cout << "\"npr valor\" número de procesadores esclavos (0)"<< endl;
            cout << "\"pop valor\" tamaño de poblacion (100)" << endl;
            cout << "\"crm valor\" tamaño del cromosoma (127)"<< endl;
            cout << "\"gen valor\" máximo de generaciones (500)"<< endl;
            cout << "\"cru valor\" probabilidad de cruza (0.9)"<< endl;
            cout << "\"mut valor\" probabilidad de mutacion (0.05)"<< endl;
            cout << "\"sel valor\" tipo de seleccion (1=RULETA, 2=competencia, 3=ventanas)"<< endl;
            cout << "\"bre valor\" tamaño de brecha generacional"<< endl;
            cout << "\"tmu valor\" tipo de mutacion (1=de cromosoma, 2 = de BIT)"<< endl;
            cout << "\"nsu valor\" cantidad de sub-poblaciones (0 desactiva)"<< endl;
            cout << "\"tsu valor\" tamanio de sub-poblaciones"<< endl;
            cout << "\"gsu valor\" maximo de generaciones para las sub-poblaciones"<< endl;
            cout << "\"cfg <cadena>\" nombre de archivo de configuracion"<< endl;
            cout << "\"help\" esta ayuda."<< endl;

            return 0;
        }
        
        if (aux == "cfg") if (argc>=(i+1)) cfg_settings = argv[i+1];

    }    
    
    //==================================
    // LEVANTO ARCHIVO DE CONFIGURACION
    //----------------------------------
        
    Dictionary SETTINGS(cfg_settings.c_str());
    //==================================
    
    //==================================
    // INICIALIZO MEDIDAS
    //----------------------------------
    
    Measures MEASURES(SETTINGS.get_int("Gmax")+1);  // generacion 0 (pob inicial) + Gmax generaciones
    //==================================
    
    cmd = SETTINGS.get_str("cmd");
    cmd.erase(std::remove(cmd.begin(), cmd.end(), '\n'), cmd.end());
    
    
    string testbin = SETTINGS.get_str("testbin");
    testbin.erase(std::remove(cmd.begin(), cmd.end(), '\n'), cmd.end());    
    if (testbin.compare("None") == 0) testbin = "./testsvm";
    

    time_t rawtime;
    struct tm * timeinfo; time ( &rawtime );
    timeinfo = localtime ( &rawtime );
  
    AG AlgGen;
    
    char * fecha = asctime(timeinfo);
    fecha[strlen(fecha)-1] = '\0';
    for (unsigned count = 0; count < strlen(fecha); count++)
       if (fecha[count] == ' ')  fecha[count] = '_';
    for (unsigned count = 0; count < strlen(fecha); count++)
       if (fecha[count] == ':')  fecha[count] = '.';   
    char faux[strlen(fecha)];
    for (unsigned count = 0; count < strlen(fecha); count++)
       faux[count] = fecha[count];
    
    fecha[0]=faux[4];fecha[1]=faux[5];fecha[2]=faux[6];
    fecha[4]=faux[8];fecha[5]=faux[9];fecha[6]=faux[7];
    fecha[7]=faux[0];fecha[8]=faux[1];fecha[9]=faux[2];    
    
    AlgGen.fecha = fecha;
    AlgGen.slvbin = cmd;
 
    AlgGen.nObjctvs = SETTINGS.get_int("NObjetivos");
    
    short count = 0;
    float last_max = 0;
    statistics result;
  
    short nproc = SETTINGS.get_int("NProcesos"); 
    short maxgen = SETTINGS.get_int("Gmax");
    
    short popsize = SETTINGS.get_int("Nindividuos");//8; //100;
    short cromsize = SETTINGS.get_int("Ngenes");//16063;      // GCM
    float tasa_activ = SETTINGS.get_dbl("TasaActivacionInicial");

    double  pmuta = SETTINGS.get_dbl("pm");//0.15/cromsize; // 0.000015; 
    short   tmuta = SETTINGS.get_int("OpMutacion");//2; // 2 = mutacion de bit
    double pcruza = SETTINGS.get_dbl("px");//0.85;
    short  brecha = SETTINGS.get_int("Brecha");//0
    short  tselec = SETTINGS.get_int("OpSeleccion");//2,
    AlgGen.Elite = SETTINGS.get_int("E"); 
    
    short  steady = SETTINGS.get_int("steady");
    // double fitmax = SETTINGS.get_dbl("fitmax");

    // Mutacion con decaimiento ->
    bool muta_expo   = false;
    bool muta_amorti = false;
    muta_expo   = SETTINGS.get_bool("Exponencial");
    muta_amorti = SETTINGS.get_bool("Amortiguada");
    
    float A_am    = SETTINGS.get_dbl("amA");
    float f_am    = SETTINGS.get_dbl("amF");
    float phi_am  = SETTINGS.get_dbl("amPhi");
   
    double gamma_ini=0;
    double gamma_fin=0;
    if (muta_expo) {
         gamma_ini = SETTINGS.get_dbl("GammaINI");
         gamma_fin = SETTINGS.get_dbl("GammaFIN");
    }      
        
    // PARAMETROS PARA FUNCION DE FITNESS SHARING    
    AlgGen.sigma_share = SETTINGS.get_dbl("SigmaShare");
    AlgGen.alfa_share  = SETTINGS.get_dbl("AlfaShare");
    AlgGen.dist_opt = SETTINGS.get_int("dist_opt");
    bool onlyBest = false;
    // onlyBest = SETTINGS.get_bool("onlyBest");
    AlgGen.activ_rate_sp = SETTINGS.get_dbl("TasaActivacionSubPob");
    AlgGen.ModifyRepeated = SETTINGS.get_bool("ModifyRepeated");
    AlgGen.FitnessOption = SETTINGS.get_int("FitnessOption");
    AlgGen.FitnessNRScale = SETTINGS.get_bool("FitnessNRScale");
    AlgGen.Rfun = SETTINGS.get_int("Rfun");
    bool yaml = true;
    yaml = SETTINGS.get_bool("WriteYamlFile");
    
    AlgGen.stepped_activ = SETTINGS.get_bool("TasaActivacionEscalonada");
    if (AlgGen.stepped_activ) {
        AlgGen.activ_rates.resize(3);
        AlgGen.activ_rates[0]=SETTINGS.get_dbl("TasaActivacion55");
        AlgGen.activ_rates[1]=SETTINGS.get_dbl("TasaActivacion30");
        AlgGen.activ_rates[2]=SETTINGS.get_dbl("TasaActivacion15");
    }
   
    // parametros para SUBPOBLACIONES ->       
    short Nsubpobs = SETTINGS.get_int("NSubPoblaciones");//0; //4; 
    short tamSubPob = SETTINGS.get_int("Nindividuos_s");//20;
    short NGenSubPob = SETTINGS.get_int("Gmax_s");//50;
    short nGenSinSubpob = SETTINGS.get_int("SPobWait");//50;    
    //     <-    
    short ngs = 0; 
    
    double y_am,t_am; 

    srand((unsigned)time(0));
    
    for (int i=1;i<argc;i++)
    {
        aux = argv[i];

        if (aux == "npr") {
            if (argc>=(i+1)) {
               nproc = atoi(argv[i+1]);
               if (nproc == 0) nproc = 1;
            }
        }

        if (aux == "pop") if (argc>=(i+1)) popsize = atoi(argv[i+1]);
        if (aux == "crm") if (argc>=(i+1)) cromsize = atoi(argv[i+1]);
        if (aux == "gen") if (argc>=(i+1)) maxgen = atoi(argv[i+1]);
        if (aux == "cru") if (argc>=(i+1)) pcruza = atof(argv[i+1]);
        if (aux == "mut") if (argc>=(i+1)) pmuta = atof(argv[i+1]);
        if (aux == "bre") if (argc>=(i+1)) brecha = atoi(argv[i+1]);
        if (aux == "sel") if (argc>=(i+1)) tselec = atoi(argv[i+1]);
        if (aux == "tmu") if (argc>=(i+1)) tmuta = atoi(argv[i+1]);
        if (aux == "nsu") if (argc>=(i+1)) Nsubpobs = atoi(argv[i+1]);
        if (aux == "tsu") if (argc>=(i+1)) tamSubPob = atoi(argv[i+1]);
        if (aux == "gsu") if (argc>=(i+1)) NGenSubPob = atoi(argv[i+1]);
        if (aux == "vrb") AlgGen.verbose = true;

    }    
       
    // DEFINO EL NOMBRE DEL ARCHIVO DE RESULTADOS
    string filename;

    filename = SETTINGS.get_str("Filename");    
    AlgGen.folder = SETTINGS.get_str("outdir");
    
    if (AlgGen.folder.compare("None") == 0){
        AlgGen.folder=".";
    }  else {
        cmd = "mkdir -p "+AlgGen.folder;
        bar = system(cmd.c_str());
        
    }       
    if (Nsubpobs>0)
         AlgGen.folder = AlgGen.folder+"/MOELIGA+Subpobs_"+fecha;          
    else
         AlgGen.folder = AlgGen.folder+"/MOELIGA_"+fecha;
     
    cmd = "mkdir -p "+AlgGen.folder;
    bar = system(cmd.c_str());
    
    
    cmd = "cp "+cfg_settings+" "+AlgGen.folder+"/";
    bar = system(cmd.c_str());
    
    // AlgGen.string_aux_filename = "_(gammas:_"+tostr(gamma_ini)+"_"+tostr(gamma_fin)+")";
    AlgGen.string_aux_filename = "";
    
    if (filename.compare("None") == 0){
        
        if (Nsubpobs>0)
            filename = AlgGen.folder+"/"+"MOELIGA+Subpobs_results_"+AlgGen.fecha+ AlgGen.string_aux_filename +".json";
        else
            filename = AlgGen.folder+"/"+"MOELIGA_results_"+AlgGen.fecha+ AlgGen.string_aux_filename +".json";
    }    
    
    if (maxgen==0) maxgen = 1700;
    if (popsize==0) popsize = 100;
    if (cromsize==0) cromsize = 127;
    
    AlgGen.gen = 0;
    AlgGen.maxgen = maxgen;
    AlgGen.nsubpob = Nsubpobs;

    /*--------------------------------------------*/
    /*          INICIALIZACION DE MPI             */

    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    /*--------------------------------------------*/

    global_tic();
    
    // INICIALIZACION DEL ALGORITMO GENETICO
    AlgGen.inicializar(popsize, cromsize , maxgen, pcruza, pmuta, nproc, tasa_activ, cfg_settings);
    result = AlgGen.estadisticas(AlgGen.pobvieja);    
            
    if (yaml) AlgGen.yaml_ImprimirGen(AlgGen.gen, result.maxfitness, result.minfitness, result.prom, AlgGen.pobvieja);
    else  AlgGen.ImprimirGen(AlgGen.gen, result.maxfitness, result.minfitness, result.prom, AlgGen.pobvieja);
    AlgGen.ImprimirFrente(AlgGen.pobvieja, AlgGen.gen, result.maxfitness, onlyBest, yaml);
    
    MEASURES.Update(AlgGen.pobvieja, AlgGen.gen); // ACTUALIZO MEDIDAS
    MEASURES.Save(filename, SETTINGS);
    
    do {

        ngs++;
        
        if (!AlgGen.verbose)
        cout << "\r" << "Generation: " << (AlgGen.gen+1);
        
        // *******************************************************//
        // generación(tamaño_brecha, tipo_selección, tipo_mutación);
        // tipo_selección: 1 = ruleta, 2 = competencia, 3 = ventanas
        // tipo_mutación: 1 = de cromosoma, 2 = de bit
        // *******************************************************//

        if (muta_expo)
        {
            // pmuta = (gamma_ini/cromsize)*pow( pow( ( (gamma_fin/cromsize) / (gamma_ini/cromsize) ), (double) (1.0/maxgen)) , (double) AlgGen.gen );            
            pmuta = gamma_ini*pow( pow( ( gamma_fin / gamma_ini ), (double) (1.0/maxgen)) , (double) AlgGen.gen ); // divido por cromsize despues, segun que tipo de mutacion utilizo   
        }
            
        if (muta_amorti)
        {        
            t_am = ((double) AlgGen.gen) / maxgen;
            y_am = A_am*pmuta*sin(2*M_PI*f_am*t_am+phi_am);        
            pmuta = y_am+pmuta; 
        }  
        
        AlgGen.gen = AlgGen.gen + 1;            
        AlgGen.generacion(brecha, tselec, tmuta, pmuta, nproc);
        
        if ((Nsubpobs>0) && (tamSubPob>0) && (NGenSubPob>0)) {
             if (ngs >= nGenSinSubpob) {
                 AlgGen.EvoSubPobs(brecha, tselec, tmuta, pmuta, nproc, Nsubpobs, tamSubPob, NGenSubPob, cfg_settings);
                 ngs = 1;
             }             
        }    

        result = AlgGen.estadisticas(AlgGen.pobnueva);
        
        // se controla desde SETTINGS con parametro steady
        if (last_max == result.maxfitness)
        {  
               count = count + 1; 
            
        } else 
        {  
               count = 0; 
        }
        
        if (yaml) AlgGen.yaml_ImprimirGen(AlgGen.gen, result.maxfitness, result.minfitness, result.prom, AlgGen.pobnueva);
        else AlgGen.ImprimirGen(AlgGen.gen, result.maxfitness, result.minfitness, result.prom, AlgGen.pobnueva);
        
        AlgGen.ImprimirFrente(AlgGen.pobnueva, AlgGen.gen, result.maxfitness, onlyBest, yaml);
        
        last_max = result.maxfitness;

        AlgGen.pobvieja = AlgGen.pobnueva;
        
        //--------------------
        // ACTUALIZO MEDIDAS
        //--------------------

        MEASURES.Update(AlgGen.pobvieja, AlgGen.gen);         
        MEASURES.Save(filename, SETTINGS);

         
    } while ((AlgGen.gen < AlgGen.maxgen) & (count<( (short) AlgGen.maxgen* (((float) steady) / 100)) ));
    if (!AlgGen.verbose) cout << endl;
    
    /* // Deprecated
     * // hago el plot de la corrida
    if (!yaml) {
       cmd = "python3 Plot4MOELIGA.py ";
       cmd.insert(cmd.length(), filename); 
       bar = system(cmd.c_str());
    } 
    */
    
    // ejecuto el test con los cromosomas del Frente 
    cmd = testbin.c_str();
    cmd.insert(cmd.length(), " file "); 
    cmd.insert(cmd.length(), AlgGen.crom_file); 
    cmd.insert(cmd.length(), " cfg "); 
    cmd.insert(cmd.length(), cfg_settings.c_str()); 
    cmd.insert(cmd.length(), " > ");     
    filename.replace(filename.find(".json"),5,".test");
    cmd.insert(cmd.length(), filename); 
    bar = system(cmd.c_str());

    /*
    cmd = "rm -fr ";
    cmd.insert(cmd.length(), AlgGen.crom_file); 
    system(cmd.c_str());
    */
    
    if (AlgGen.verbose) 
    {
        cmd = "cat ";
        cmd.insert(cmd.length(), filename); 
        bar = system(cmd.c_str());
    }
    
    AlgGen.Terminar(nproc,AlgGen.pobvieja.lcrom,!yaml);
    
    (void) bar;

    return 0;

}
//################################################
// ------------- END MAIN FUNCTION --------------
//################################################
