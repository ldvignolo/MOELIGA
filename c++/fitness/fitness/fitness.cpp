//----------------------------------------------------------------------------
//
//  Funcion de Fitness 
//
//
// 
//  17/06/20 -> Classifiers SVM and LVM + Relief Measure
// 
//  LDV
//
//----------------------------------------------------------------------------


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <float.h>
#include <vector>
#include <cctype>
#include <ctime>

#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR

#include <mpi.h>

#include "../../GA/types.h"
#include "../../GA/utils.h"
#include "../../configs/Toolbox.hpp"   // LECTURA DEL ARCHIVO DE CONFIGURACION

//--------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "../libsvm-3.20/svm.h"
#include "svm-train.c"
#include "svm-predict.c"
#include "arffread.c"
#include "../SLFN/elm.cpp"

//--------------------------------------

#include <unistd.h>
#include <ios>



using namespace std;


struct svm_parameter params;
struct svm_problem trnD,tstD,auxD;
struct svm_node *trn_space, *tst_space, *aux_space;
bool normalizar=true, data_shuffle=true;

const char* trnfile="";
const char* tstfile="";

string configs_trn = "";
string configs_tst = "";
string clasificador = "svm";
string str_svm = "SVM";
string str_elm = "ELM";

struct t_elm_par{
    int nhn;
    double rf;
    bool multi;
    int nhn_max;
};    
t_elm_par elm_params;

vector <double> fitness(cromosoma crom, int lbits, int rank, float seed, short pobtype, int NObjectives, float ptrain, unsigned ntest, bool c_eval);
vector <string> SplitWords(string strString);
struct svm_model* train(unsigned Nfeat, struct svm_problem datos);
double test(string configs, struct svm_problem datos, struct svm_model *modelo, vector <int> labels);
void process_mem_usage(double& vm_usage, double& resident_set);
void scale_data(unsigned Ncols);
void scale_data(unsigned Ncols, cromosoma crom);
void make_partition(unsigned Nfeat);
double distL1(svm_node *x, svm_node *y, int Nf);
double distL1(svm_node *x, svm_node *y, vector <int> indF);
double Rmeasure(struct svm_problem data, int Nf);
double Rmeasure(struct svm_problem data, cromosoma crom);
double elm(struct svm_problem trn_data, struct svm_problem tst_data, int Nfeats, int elm_nhn, double elm_rf, bool multi, int max_nhn);


double sigmoid(double value, double lambda)
{
    return  1.0 / (1.0 + exp( (-1.0)*value*lambda ) );
}    


int main(int argc, char** argv)
{
    int nlcrom,lcrom = 4;   // valor para inicializar, recibo valor verdadero por MPI
                          
    int id = atoi(argv[1]);
    cromosoma cromovect;
    vector <double> aptitud;
    int i, ini_NObjectives=1, NObjectives; 
    int params[3];
    float seed=0;
    short pobtype=0;
    bool c_eval=true;

    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    MPI_Status status;
    MPI_Comm parent_comm;
    MPI_Comm_get_parent(&parent_comm);

    int* cromo = (int*) malloc(lcrom * sizeof(int));
    double* fit = (double*) malloc(ini_NObjectives * sizeof(double));
    
    cromo[0]=10;
        
    //==================================
    // LEVANTO ARCHIVO DE CONFIGURACION
    //----------------------------------
    
    Dictionary SETTINGS(argv[2]);

    string aux1 = "", aux2 = "";
    aux1 = SETTINGS.get_str("trnfile");
    trnfile=aux1.c_str();
    aux2 = SETTINGS.get_str("tstfile");
    tstfile=aux2.c_str();
    
    configs_trn = SETTINGS.get_str("configs_trn");
    configs_tst = SETTINGS.get_str("configs_tst");       
    
    configs_trn.insert(configs_trn.begin(),' ');
    configs_trn.insert(configs_trn.end(),' ');
    configs_tst.insert(configs_tst.begin(),' ');
    configs_tst.insert(configs_tst.end(),' ');         
    
    elm_params.nhn = SETTINGS.get_int("elm_nhn");  
    elm_params.rf  = SETTINGS.get_dbl("elm_rf");
    elm_params.multi = SETTINGS.get_bool("elm_nhn_X_nf");      
    elm_params.nhn_max = SETTINGS.get_bool("elm_nhn_max");    
    
    clasificador = SETTINGS.get_str("classifier");
    
    bool Obj2Sigmod = SETTINGS.get_bool("Obj2Sigmod"); // false por omision     
    float SigmLambda = SETTINGS.get_dbl("SigmLambda"); 
    if (SigmLambda>500.0) SigmLambda = 2.5; // valor por omision
    
    float ptrain = SETTINGS.get_dbl("ptrain");
    unsigned ntest = SETTINGS.get_int("NTests");
    if (ntest<=0) ntest=1;
    
    if ( (ptrain<=0) || (ptrain>100) ) ptrain = 100;
    
    // configs_trn=" -s 0 -t 0 -q ";
    // configs_tst=" -q ";                
    // cout << "-> " << configs_trn << endl;
    // cout << "-> " << configs_tst << endl;

    /*--------------------------------*/

    unsigned Nfeat;
    ReadProblemArff(trnfile, Nfeat, trnD, &trn_space[0]);
    ReadProblemArff(tstfile, Nfeat, tstD, &tst_space[0]);
    
    /*--------------------------------------------------*/
    /*--------------------------------------------------*/
    
    auxD.y  = Malloc(double,trnD.l+tstD.l);
    auxD.x  = Malloc(struct svm_node *,trnD.l+tstD.l);
    auxD.l  = trnD.l+tstD.l;
    aux_space = Malloc(struct svm_node,(Nfeat+1) * auxD.l); 
    unsigned kk = 0;

    for (int ii=0;ii<trnD.l;ii++){
        auxD.x[ii]  = &aux_space[kk];        
        for (unsigned jj=0;jj<Nfeat;jj++){
            aux_space[kk].index = trnD.x[ii][jj].index;
            aux_space[kk].value = trnD.x[ii][jj].value;
            kk++; 
        }    
        aux_space[kk].index = -1;
        auxD.y[ii]  =  trnD.y[ii];
        kk++;
    }

    for (int ii=0;ii<tstD.l;ii++){
        auxD.x[trnD.l+ii]  = &aux_space[kk];        
        for (unsigned jj=0;jj<Nfeat;jj++){
            aux_space[kk].index = tstD.x[ii][jj].index;
            aux_space[kk].value = tstD.x[ii][jj].value;
            kk++;
        }    
        aux_space[kk].index = -1;
        auxD.y[trnD.l+ii]  =  tstD.y[ii];
        kk++;
    }    

    srand ( unsigned ( std::time(0) ) );
    
    /*--------------------------------------------------*/
    /*--------------------------------------------------*/
     
    normalizar = SETTINGS.get_bool("Normalizar");  
    data_shuffle = SETTINGS.get_bool("DataShuffle");  
     
    // cout << "<-> " << rank <<" -> datos leidos " <<endl;
    
    /*cout << trnfile << endl;
    cout << tstfile << endl;
    cout << configs_trn << endl;
    cout << configs_tst << endl;*/
        
    vector <string> cadena;
    cadena = SplitWords(configs_trn);
    int Xargc = cadena.size();
    char** Xargv = Malloc(char*,Xargc);
    for (int i=0;i<Xargc;i++){
       Xargv[i] = Malloc(char,cadena[i].length());
       strcpy(Xargv[i], cadena[i].c_str()); 
    }
    parse_command_line(Xargc, Xargv);
     
    // MPI_Recv(params, 2, MPI_INTEGER, 0, id, parent_comm, &status);
    MPI_Recv(&seed, 1, MPI_FLOAT, 0, id, parent_comm, &status);
    MPI_Recv(&NObjectives, 1, MPI_INTEGER, 0, id, parent_comm, &status);     
    
    if (!data_shuffle) {
        make_partition(Nfeat);               
        if (normalizar) scale_data(Nfeat);
    }
    
    /*--------------------------------*/
    unsigned cc = 0;
    while (cromo[0]!=-5)
    {       
        cc++;
        
        MPI_Recv(params, 3, MPI_INTEGER, 0, id, parent_comm, &status);

        nlcrom  = params[0];
        pobtype = params[1];
        c_eval  = params[2]!=0; 
        
        if (nlcrom!=lcrom){
            lcrom = nlcrom;
            cromo = (int*) realloc(cromo, lcrom * sizeof(int)); 
        }
        
        if (ini_NObjectives!=NObjectives){
            ini_NObjectives=NObjectives;
            fit = (double*) realloc(fit, NObjectives * sizeof(double));
        }

        MPI_Recv(cromo, lcrom, MPI_INTEGER, 0, id, parent_comm, &status);
        
        if (cromo[0]!=-5)
        {
        
           cromovect.resize(0);
           for (i=0;i<lcrom;i++){
               if (cromo[i]==1) 
                 cromovect.push_back(true);
               else 
                 cromovect.push_back(false);
           }
        
           aptitud = fitness(cromovect, lcrom, rank, seed, pobtype, NObjectives, ptrain, ntest, c_eval); 
           
           if ((Obj2Sigmod) && (1<NObjectives))
               aptitud[1] = sigmoid(aptitud[1], SigmLambda); 
           
           for (i=0;i<NObjectives;i++) fit[i] = aptitud[i];
           MPI_Send(fit, NObjectives, MPI_DOUBLE, 0, id, parent_comm);
        }
        
    }
    
    free(cromo);
    free(fit);
    free(Xargv);
    free(trnD.y);
    free(tstD.y);
    free(tst_space);
    free(trn_space);
    free(aux_space);
    
    MPI_Finalize();
    
    return 0;

}





double distL1(svm_node *x, svm_node *y, vector <int> indF)
{
    double dist = 0.0;
    for (int i=0;i<indF.size();i++)
        dist = dist + fabs(x[indF[i]].value-y[indF[i]].value);
    dist = dist / indF.size();
    
    return dist;
}

double distL1(svm_node *x, svm_node *y, int Nf)
{
    double dist = 0.0;
    for (int i=0;i<Nf;i++)
        dist = dist + fabs(x[i].value-y[i].value);
    dist = dist / Nf;
    
    return dist;
}


double Rmeasure(struct svm_problem data, cromosoma crom)
{
    vector <int> indF;
    for (int i=0;i<crom.size();i++) 
        if (crom[i]) indF.push_back(i);

    int Nd = data.l;    
    // int Nr = ceil(0.2*Nd);
    int Nr = ceil(0.35*Nd);
    
    vector <int> indI;
    
    for (int i=0;i<Nd;i++) indI.push_back(i);
    random_shuffle ( indI.begin(), indI.end() );
    
    double nmiss, nhit, dist, measure = 0.0;
    
    for (unsigned k=0;k<Nr;k++){
        
        nmiss = __DBL_MAX__;
        nhit = __DBL_MAX__;        
        for (unsigned j=0;j<Nd;j++){
            if (j!=indI[k]) 
            {                
                dist = distL1(data.x[j], data.x[indI[k]], indF);
                if (data.y[j] == data.y[indI[k]]) {
                    if (dist < nhit) nhit = dist;                    
                } else {
                    if (dist < nmiss) nmiss = dist;                
                }    
            }
        }
        measure = measure - nhit + nmiss;        
    }
    measure = measure / Nr;
    
    return measure;
}

double Rmeasure(struct svm_problem data, int Nf)
{

    int Nd = data.l;    
    int Nr = ceil(0.2*Nd);
    
    vector <int> indI;
    
    indI.resize(Nd);
    for (int i=0;i<Nd;i++) indI[i]=i;
    random_shuffle ( indI.begin(), indI.end() );
    
    double nmiss, nhit, dist, measure = 0.0;
    for (unsigned k=0;k<Nr;k++){
        
        nmiss = DBL_MAX;
        nhit = DBL_MAX;        
        for (unsigned j=0;j<Nd;j++){
            if (j!=indI[k]) 
            {                
                dist = distL1(data.x[j], data.x[indI[k]], Nf);
                if (data.y[j] == data.y[indI[k]]) {
                    if (dist < nhit) nhit = dist;
                } else {
                    if (dist < nmiss) nmiss = dist;                
                }               
            }
        }
        measure = measure - nhit + nmiss;        
    }
    measure = measure / Nr;
    
    return measure;
}





void make_partition(unsigned Nfeat)
{

    unsigned trnN = trnD.l, tstN = tstD.l;

    free(trnD.y);
    free(tstD.y);
    free(tst_space);
    free(trn_space);
    
    trnD.y  = Malloc(double,trnN);
    trnD.x  = Malloc(struct svm_node *,trnN);
    tstD.y  = Malloc(double,tstN);
    tstD.x  = Malloc(struct svm_node *,tstN);
    trnD.l = trnN; tstD.l = tstN;
    trn_space = Malloc(struct svm_node,trnD.l*(Nfeat+1));
    tst_space = Malloc(struct svm_node,tstD.l*(Nfeat+1));
        
    vector <unsigned> rndidx;
    for (unsigned i=0; i<trnN+tstN; i++) rndidx.push_back(i);
    random_shuffle ( rndidx.begin(), rndidx.end() );
    
    unsigned kk = 0;
    for (int ii=0;ii<trnD.l;ii++){
        trnD.x[ii]  = &trn_space[kk];        
        for (unsigned jj=0;jj<Nfeat;jj++){
            trn_space[kk].index = auxD.x[rndidx[ii]][jj].index;
            trn_space[kk].value = auxD.x[rndidx[ii]][jj].value;
            kk++; 
        }    
        trn_space[kk].index = -1;
        trnD.y[ii]  = auxD.y[ii];
        kk++;
    }

    kk = 0; 
    for (int ii=0;ii<tstD.l;ii++){
        tstD.x[ii]  = &tst_space[kk];        
        for (unsigned jj=0;jj<Nfeat;jj++){
            tst_space[kk].index = auxD.x[rndidx[trnD.l+ii]][jj].index;
            tst_space[kk].value = auxD.x[rndidx[trnD.l+ii]][jj].value;
            kk++;
        }    
        tst_space[kk].index = -1;
        tstD.y[ii]  =  auxD.y[trnD.l+ii];
        kk++;
    }  

}



void scale_data(unsigned Ncols)
{

     /*
     vector <double> vmin, vmax;      
     for (unsigned k=0;k<Ncols;k++)
     {
        vmin.push_back(__DBL_MAX__);
        vmax.push_back(__DBL_MIN__);
        for (int i=0;i<trnD.l;i++)
        {   
            if (vmin[k]>trnD.x[i][k].value) vmin[k]=trnD.x[i][k].value;
            if (vmax[k]<trnD.x[i][k].value) vmax[k]=trnD.x[i][k].value;            
        }        
     }
     */
     
     vector <double> vmean, vstd;
     for (unsigned k=0;k<Ncols;k++)
     {
         vmean.push_back(0.0);
         for (int i=0;i<trnD.l;i++) vmean[k]=vmean[k]+trnD.x[i][k].value;
     }
     
     for (unsigned k=0;k<Ncols;k++)
     {
         vmean[k]=vmean[k]/trnD.l;
         vstd.push_back(0.0);
         for (int i=0;i<trnD.l;i++) vstd[k]=vstd[k]+pow(trnD.x[i][k].value - vmean[k],2.0);
     }
     for (unsigned k=0;k<Ncols;k++)
             vstd[k]=sqrt(vstd[k]/trnD.l);

     for (int i=0;i<trnD.l;i++)
         for (unsigned k=0;k<Ncols;k++)
             trnD.x[i][k].value = (trnD.x[i][k].value - vmean[k]) / vstd[k];
             // trnD.x[i][k].value = ( (trnD.x[i][k].value - vmin[k]) - (vmax[k] - vmin[k])/2 )  / (vmax[k] - vmin[k]);
             // trnD.x[i][k].value = 2.0*( (trnD.x[i][k].value - vmin[k]) / (vmax[k] - vmin[k]) - 0.5 );

     
     for (int i=0;i<tstD.l;i++)
         for (unsigned k=0;k<Ncols;k++)
             tstD.x[i][k].value = (tstD.x[i][k].value - vmean[k]) / vstd[k];             
             // tstD.x[i][k].value = ( (tstD.x[i][k].value - vmin[k]) - (vmax[k] - vmin[k])/2 )  / (vmax[k] - vmin[k]);
             // tstD.x[i][k].value = 2.0*( (tstD.x[i][k].value - vmin[k]) / (vmax[k] - vmin[k])  - 0.5 );

}



void scale_data(unsigned Ncols, cromosoma crom)
{
 
     vector <int> index;
     for (unsigned k=0;k<Ncols;k++)
         if (crom[k]) index.push_back(k);
     int NF = index.size();    
     
     vector <double> vmean, vstd;
     for (unsigned k=0;k<NF;k++)
     {
         vmean.push_back(0.0);
         for (int i=0;i<trnD.l;i++) vmean[k]=vmean[k]+trnD.x[i][index[k]].value;
     }
     
     for (unsigned k=0;k<NF;k++)
     {
         vmean[k]=vmean[k]/trnD.l;
         vstd.push_back(0.0);
         for (int i=0;i<trnD.l;i++) vstd[k]=vstd[k]+pow(trnD.x[i][index[k]].value - vmean[k],2.0);
     }
     for (unsigned k=0;k<NF;k++)         
             vstd[k]= sqrt(vstd[k]/trnD.l);

     for (int i=0;i<trnD.l;i++)
         for (unsigned k=0;k<NF;k++)
             trnD.x[i][index[k]].value = (trnD.x[i][index[k]].value - vmean[k]) / vstd[k];
   
     for (int i=0;i<tstD.l;i++)
         for (unsigned k=0;k<NF;k++)
              tstD.x[i][index[k]].value = (tstD.x[i][index[k]].value - vmean[k]) / vstd[k];

}



void process_mem_usage(double& vm_usage, double& resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   //
   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize / (1024.0*1024.0);
   resident_set = rss * page_size_kb/ 1024;
}



vector <double> fitness(cromosoma crom, int lbits, int rank, float seed, short pobtype, int NObjectives, float ptrain, unsigned ntest, bool c_eval)
{
     string cadena, cadena1, aux;
     vector <double> aptitude;
     aptitude.resize(NObjectives);
     int Lcrom = crom.size();
     
     /*double vm, rss;
     process_mem_usage(vm, rss);
     printf("-> fitsvm 1: %f\n ",rss);
     */
     
     if (Lcrom!=lbits)
     {
        cout << ">> Error en el tamaño del cromosoma <<" << endl;
        aptitude.resize(1);
        aptitude[0]=-1;
        return aptitude;
     }

     int CFeats = 0;
     for (int k=0;k<Lcrom;k++)
     {  
         if (crom[k])  CFeats++;
     }
     
     for (short k=0; k<NObjectives; k++) aptitude[k] = 0.0;
     
     if (CFeats==0){
      
        aptitude[1] = 1.0;        
        aptitude[0] = 0.0;
        
        if (NObjectives > 2)
           aptitude[2] = 0.0;
             
        return aptitude;         
     }
     
     /************************************************************************************/
     
     size_t elements;
     struct svm_model *modelo;
     struct svm_problem trnD_aux, tstD_aux;
     struct svm_node *trn_space_aux, *tst_space_aux;
     
     int j;          
     
     double fit_aux, mR;
     for (unsigned jk=0;jk<ntest;jk++){        
         
         int idx, Ncols = CFeats+1;
         
         int Ntrn = (int) trnD.l*(ptrain/100);
         if ((Ntrn>=0) || (Ntrn>trnD.l)) Ntrn=trnD.l;
         
         trnD_aux.l = Ntrn;              // number of rows
         elements = Ncols * Ntrn;   // total number of features (nfeat * prob.l)     
         trnD_aux.y  = Malloc(double,trnD_aux.l);
         trnD_aux.x  = Malloc(struct svm_node *,trnD_aux.l);
         trn_space_aux = Malloc(struct svm_node,elements);
         
         tstD_aux.l = tstD.l;              // number of rows
         elements = Ncols * tstD.l;   // total number of features (nfeat * prob.l)     
         tstD_aux.y  = Malloc(double,tstD_aux.l);
         tstD_aux.x  = Malloc(struct svm_node *,tstD_aux.l);
         tst_space_aux = Malloc(struct svm_node,elements);          
         
         if (data_shuffle) {
            make_partition(Lcrom);               
            if (normalizar) scale_data(Lcrom, crom);
         }
     
         vector <short> pindx; 
         for (int i=0; i<trnD.l; i++) pindx.push_back(i);
         random_shuffle ( pindx.begin(), pindx.end() );
         
         j=0;
         for (int i=0;i<Ntrn;i++)
         {
             trnD_aux.x[i] = &trn_space_aux[j];           
             trnD_aux.y[i] = trnD.y[pindx[i]];
             idx = 1;
             for (int k=0;k<Lcrom;k++)
             {   
                 if (crom[k]){
                 trn_space_aux[j].index = idx;
                 trn_space_aux[j].value = trnD.x[pindx[i]][k].value;
                 j++;
                 idx++;
                 }
             }    
             trn_space_aux[j].index = -1;
             j++;
         }   
         
         vector <int> tst_labels;
         j=0;
         for (int i=0;i<tstD.l;i++)
         {
             tstD_aux.x[i] = &tst_space_aux[j];           
             tstD_aux.y[i] = tstD.y[i];
             tst_labels.push_back(tstD.y[i]);
         
             idx = 1;
             for (int k=0;k<Lcrom;k++)
             {   
                 if (crom[k]){
                     tst_space_aux[j].index = idx;
                     tst_space_aux[j].value = tstD.x[i][k].value;
                     j++;
                     idx++;
                 }
             }
             tst_space_aux[j].index = -1;
             j++;
         }
         
         if (c_eval) 
         {            
            if (caseInSensStringCompare(clasificador, str_svm)) {
            
                modelo = train(CFeats, trnD_aux);
                fit_aux = test(configs_tst, tstD_aux, modelo, tst_labels);                
                svm_free_and_destroy_model(&modelo);  
                
            } else if (caseInSensStringCompare(clasificador, str_elm)) {
            
                fit_aux = elm(trnD_aux, tstD_aux, CFeats, elm_params.nhn, elm_params.rf, elm_params.multi, elm_params.nhn_max);
            }
            
         } else fit_aux = 0.0;
         
         aptitude[0] = aptitude[0] + fit_aux;
         
         if (NObjectives > 2) {
            mR = Rmeasure(tstD_aux, CFeats);         
            aptitude[2] = aptitude[2] + mR;
         }
                 
         free(trnD_aux.y);
         free(trn_space_aux);
         free(tstD_aux.y);
         free(tst_space_aux);
     
     }
     
     aptitude[0] = aptitude[0]/ntest;
     if (NObjectives > 2)
         aptitude[2] = aptitude[2]/ntest;
       
     if (Lcrom!=0) 
     {
        aptitude[1] = (double (Lcrom-CFeats))/Lcrom;        
        // aptitude[1] = sigmoid(aptitude[1], 2.5);  por simplicidad se aplica en main segun SETTINGS     
     } else 
        aptitude[1] = 0;

     
     return aptitude;
     
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

  while(i < strString.length())
  {    
    if(isspace(strString.at(i)))
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


struct svm_model* train(unsigned Nfeat, struct svm_problem datos)
{
    const char *error_msg=NULL;

    // struct svm_model *modelo=NULL;
    struct svm_model *modelo = Malloc(svm_model,1);
    
        if(param.kernel_type == PRECOMPUTED)
        for(int i=0;i<datos.l;i++)
        {
            if (datos.x[i][0].index != 0)
            {
                fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
                exit(1);
            }
            if ((int)datos.x[i][0].value <= 0 || (unsigned)datos.x[i][0].value > Nfeat)
            {
                fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
                exit(1);
            }
        }    
        
        if(param.gamma == 0 && Nfeat > 0)
              param.gamma = 1.0/Nfeat;
    
    error_msg = svm_check_parameter(&datos,&param);

    if(error_msg)
    {
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
    }        

    if(cross_validation)
    {
        do_cross_validation();
    }
    else
    {
        // modelo = svm_train(&datos,&param);
        svm_train2(&datos,&param,modelo);
    }
    
    return(modelo);
}



double test(string configs, struct svm_problem datos, struct svm_model *modelo, vector <int> labels)
{
    
    int nclass;
    vector <string> cadena;
    vector <int> pred_labels, classes;
    cadena = SplitWords(configs);
    
    int Xargc = cadena.size();
    char** Xargv = Malloc(char*,Xargc);
    for (int i=0;i<Xargc;i++){
       Xargv[i] = Malloc(char,cadena[i].length());
       strcpy(Xargv[i], cadena[i].c_str()); 
    }

    int i,k;
    // parse options
    for(i=0;i<Xargc;i++)
    {
        if(Xargv[i][0] != '-') break;
        ++i;
        switch(Xargv[i-1][1])
        {
            case 'b':
                predict_probability = atoi(Xargv[i]);
                break;
            case 'q':
                info = &print_null;
                i--;
                break;
            default:
                fprintf(stderr,"Unknown option: -%c\n", Xargv[i-1][1]);
                exit_test_help();
        }
    }

    if(predict_probability)
    {
        if(svm_check_probability_model(modelo)==0)
        {
            fprintf(stderr,"Model does not support probabiliy estimates\n");
            exit(1);
        }
    }
    else
    {
        if(svm_check_probability_model(modelo)!=0)
            info("Model supports probability estimates, but disabled in prediction.\n");
    }

    pred_labels = predict(datos, modelo);

    for (int i=0;i<Xargc;i++){
       free(Xargv[i]);
    } free(Xargv);
    
    // calcular UAR con los vectores labels y pred_labels

    vector <int> tmp;
    bool neu;
    for (i=0;i<(int)labels.size();i++){
      neu = true;     
      for (k=0;k<(int)tmp.size();k++)
          if (labels[i]==tmp[k]) { neu = false; break; }        
      if (neu) tmp.push_back(labels[i]);
    }

    nclass = tmp.size();
    int MC[nclass][nclass] = {{0}};
    memset(MC,0,nclass*nclass*sizeof(int));
      
    for (i=0;i<(int)labels.size();i++)
           MC[ labels[i]-1 ][ pred_labels[i]-1 ]++;

    double rr, UAR=0;
    for (i=0;i<nclass;i++){
       rr = 0;
       for (k=0;k<nclass;k++)
           rr = rr + MC[ i ][ k ];
       UAR = UAR + MC[ i ][ i ]/rr;
    }
    UAR = UAR/nclass;
    
    /* for (i=0;i<nclass;i++){
         for (k=0;k<nclass;k++)
           cout << MC[ i ][ k ] << " ";
         cout << "\n"; }
       cout << ">>UAR: " << UAR << endl;
    */
        
    return(UAR);
}


double elm(struct svm_problem trn_data, struct svm_problem tst_data, int Nfeats, int in_nhn, double rf, bool multi, int max_nhn) 
{

    double * xData;
    double * yData;

    int nhn;
    
    if (multi) { 
        
        nhn = Nfeats*in_nhn; 
        if (nhn > max_nhn) nhn = max_nhn;
        
    } else nhn = in_nhn;
    
    //
    // MODEL TRAINING
    //

    xData = Malloc(double,trn_data.l*Nfeats);
    yData = Malloc(double,trn_data.l);
    
    int j=0;
    for (int i=0;i<trn_data.l;i++)
    {
       for (int k=0;k<Nfeats;k++)
       {   
             xData[j] = trn_data.x[i][k].value;              
             j++;         
       }    
       yData[i] = trn_data.y[i];              
    }
    
    // train input data info: features (# of inputs) and number of samples. X must be a matrix
    int nfeatX = Nfeats;
    int nsmpX = trn_data.l;

    MatrixXd inW;
    MatrixXd bias;
    MatrixXd outW;

    // launch training procedure
    int code = elmTrain(xData, nfeatX, nsmpX,
        yData,
        nhn, rf,
        inW, bias, outW);

    if (code != 0)
        cout << "Failed to train a model.";

    //
    // MODEL TEST
    //

    xData = Malloc(double,tst_data.l*Nfeats);
    vector <double> yTest;
    
    j=0;
    for (int i=0;i<tst_data.l;i++)
    {
       for (int k=0;k<Nfeats;k++)
       {   
             xData[j] = tst_data.x[i][k].value;              
             j++;         
       }         
       yTest.push_back(tst_data.y[i]);
    }

    //test input data info: features (# of inputs) and number of samples. X must be a matrix
    nfeatX = Nfeats;
    nsmpX =  tst_data.l;

    MatrixXd mScores;

    // launch test procedure
    code = elmTest(xData, nfeatX, nsmpX,
        mScores,
        inW, bias, outW);

    if (code != 0)
        cout << "Failed to test a model.";

    //
    // SCORING
    //
    // double acc = ScoreAccuracy(mScores, yTest);
    double UAR = ScoreUAR(mScores, yTest, false);
    
    return UAR;
}
