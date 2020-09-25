//----------------------------------------------------------------------------
//
//  Función de Fitness para la versión paralela
//
//
// 
//  USAGE: ./testsvm file cromo_GCM.txt cfg gcm_SETTINGS_Test.cfg
// 
//  Leandr0
//  
//  export LD_LIBRARY_PATH="/usr/local/lib64/:$LD_LIBRARY_PATH"
//----------------------------------------------------------------------------


  /**
   *    IMPORTANTE:
   *    * LOS ARCHIVOS ARFF NO DEBEN CONTENER TABULACIONES
   *    * USAR la funcion loadarff modificada para soporte de labels strings
   */  


//--------------------------------------------
// mlpack includes

#include <mlpack/core.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/scaler_methods/mean_normalization.hpp>
// #include <mlpack/core/data/split_data.hpp>

// #include <mlpack/core/cv/k_fold_cv.hpp>
// #include <mlpack/core/data/one_hot_encoding.hpp>
// #include <mlpack/core/cv/metrics/accuracy.hpp>


#include "loadarff.hpp"
#include "methods.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::ann;
using namespace mlpack::kmeans;


//--------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <float.h>
#include <vector>
#include <cctype>

#include "../GA/types.h"
#include "../GA/utils.h"
#include "../configs/Toolbox.hpp"   // LECTURA DEL ARCHIVO DE CONFIGURACION

//--------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

//--------------------------------------

#include <unistd.h>
#include <ios>

#include <stack>
#include <ctime>


using namespace std;


vector <string> clasificadores;
vector <string> clasif_configs;
string optimizador, optim_configs;
size_t  numClasses;

arma::mat TRNdata;
arma::mat TSTdata;
arma::Row<size_t> trnLabels, tstLabels;


void test(cromosoma crom, int lbits, int rank, float seed, short pobtype, double alpha, double beta, int NObjectives);
void process_mem_usage(double& vm_usage, double& resident_set);
arma::mat MC(arma::Row<size_t> labels, arma::Row<size_t> predictedLabels);
void printMC(arma::mat MC, string offset);



int main(int argc, char** argv)
{
    cromosoma cromovect;
    int lcrom, i, ini_NObjectives=1, NObjectives; 
    float seed=0;
    short pobtype=0;

    //=========================================
    // BUSCO PARAMETROS EN LA LINEA DE COMANDOS
    //-----------------------------------------
    
    string aux, filename, cfg_settings = "SETTINGS_Test.cfg";
        
    for (int i=0;i<argc;i++)
    {
        aux = argv[i];

        if ((aux == "help")||(argc==1))
        {
            cout << "\"file <cadena>\" nombre de archivo de soluciones"<< endl;
            cout << "\"cfg <cadena>\" nombre de archivo de configuracion"<< endl;
            cout << "\"help\" esta ayuda."<< endl;

            return 0;
        }
        
        if (aux == "cfg") if (argc>=(i+1)) cfg_settings = argv[i+1];
        if (aux == "file") if (argc>=(i+1)) filename = argv[i+1];

    }    
    
    //==================================
    // LEVANTO ARCHIVO DE CONFIGURACION
    //----------------------------------
    
    Dictionary SETTINGS(cfg_settings.c_str());
          
    double alpha = SETTINGS.get_dbl("alpha");  
    double beta  = SETTINGS.get_dbl("beta");
            
    // solo por compatibilidad con settings viejos    
    string trnfile = SETTINGS.get_str("TESTFINALtrnfile");
    string tstfile = SETTINGS.get_str("TESTFINALtstfile");
    
    if (trnfile.compare("None") == 0){
        trnfile = SETTINGS.get_str("trnfile");
    }     
    if (tstfile.compare("None") == 0){
        tstfile = SETTINGS.get_str("tstfile");
    } 
    
    // cout << trnfile << endl;
    // cout << tstfile << endl;
    optimizador = SETTINGS.get_str("Optimizer");  
    string clasificador = SETTINGS.get_str("test_classifier");    
    clasificadores = SplitWords(clasificador);
    
    for (size_t i=0;i<clasificadores.size();i++)
    {
        aux = "classifier_";
        aux+=clasificadores[i];
        aux+="_config";        
        clasif_configs.push_back(SETTINGS.get_str(aux.c_str()));
    }    
    
    aux = "optimizer_";
    aux+=optimizador;
    aux+="_config";        
    optim_configs = SETTINGS.get_str(aux.c_str());
    std::transform(optimizador.begin(), optimizador.end(), optimizador.begin(), ::tolower);
    
    /*--------------------------------*/
    
    data::DatasetInfo datasetInfo;  
    data::LoadARFF(trnfile, TRNdata, datasetInfo);  
    data::LoadARFF(tstfile, TSTdata, datasetInfo);
    // we need to extract the labels from the last dimension of the dataset and remove the labels from the training set:
    trnLabels =  arma::conv_to<arma::Row<size_t>>::from(TRNdata.row(TRNdata.n_rows - 1));
    TRNdata.shed_row(TRNdata.n_rows - 1); // elimino fila correspondiente a las etiquetas  
    tstLabels  =  arma::conv_to<arma::Row<size_t>>::from(TSTdata.row(TSTdata.n_rows - 1));
    TSTdata.shed_row(TSTdata.n_rows - 1); // elimino fila correspondiente a las etiquetas
    arma::Row<size_t> uniqueLabels = arma::unique(trnLabels);
    numClasses = uniqueLabels.n_elem;
    unsigned Nfeat = TSTdata.n_rows;
    arma::mat data_aux;
    
    bool normalizar = SETTINGS.get_bool("Normalizar");        
    bool estandarizar = SETTINGS.get_bool("Estandarizar");
    bool shuffle = SETTINGS.get_bool("DataShuffle"); 
        
    if (normalizar)
    {         
       // Fit the features.
       data::MeanNormalization scale;       
       scale.Fit(TRNdata);
       // Scale the features.
       scale.Transform(TRNdata, data_aux);
       TRNdata = data_aux;  
       data_aux.clear();       
       scale.Transform(TSTdata, data_aux);
       TSTdata = data_aux; 
       data_aux.clear();         
    } 
    else if (estandarizar)
    {         
       // Fit the features.
       data::StandardScaler scale;
       scale.Fit(TRNdata);
       // Scale the features.
       scale.Transform(TRNdata, data_aux);
       TRNdata = data_aux;  
       data_aux.clear();       
       scale.Transform(TSTdata, data_aux);
       TSTdata = data_aux; 
       data_aux.clear();         
    }        
    
    if (shuffle)
    {    
       // TRAIN DATA SHUFFLE      
       arma::Row<size_t> auxLabels;    
       math::ShuffleData(TRNdata, trnLabels, data_aux, auxLabels);
       TRNdata = data_aux;  
       trnLabels = auxLabels;
       data_aux.clear();                   
       auxLabels.clear();    
    }
    
    
    vector <int> feats;
    int aux_int;
   
    NObjectives = ini_NObjectives;
    lcrom = Nfeat;
    pobtype = 0;
    
    auto doTest = [&] () {

        cromovect.resize(0);

        for (i=0;i<lcrom;i++)
             cromovect.push_back(false);

        for (unsigned r=0;r<feats.size();r++)
            if (feats[r]<lcrom) cromovect[feats[r]]=true;
       
        test(cromovect, lcrom, 0, seed, pobtype, alpha, beta, NObjectives); 

    };
    
    
    int icrom=1;
    if (argc < 4)
    {
      cout << "[" << endl;
      for (unsigned j=0;j<Nfeat;j++) feats.push_back(j);
      cout << " {\"INDIVIDUO\": " << icrom << "," << endl;      
      cout << "  \"NUMERO_DE_FEATURES\": " << feats.size() << "," << endl;
      cout << "  \"FEATURES\": " << "[";             
      for (unsigned j=0;j<feats.size();j++) 
         if (j < feats.size()-1) cout << feats[j]+1 << ", "; else cout << feats[j]+1 << "],"<< endl;      
      doTest();
      cout << " }" << endl << "]" << endl;
      
    } else
    {
        // filename = argv[1];
        ifstream features (filename.c_str());
        
        if(!features) {
            cout<<"Couldn't open the file"<<endl;
            exit(1);
        }
        
        cout << "[" << endl;
        
        string line;           
        // considero que puede haber varios cromosomas en un mismo archivo, uno por linea, entonces hago el test para cada linea
        while ( getline( features, line )) {
            
             if (icrom>1) cout << "," << endl;
            
             cout << " {\"INDIVIDUO\": " << icrom << "," << endl;
             
             feats.resize(0);
             stringstream str(line);
             
             while (str >> aux_int) {
                 aux_int = aux_int-1;
                 feats.push_back(aux_int);                 
             }             
             
             cout << "  \"NUMERO_COEFICIENTES_SELECCIONADOS\": " << feats.size() << "," << endl;
             cout << "  \"FEATURES\": " << "[";
             
             for (unsigned j=0;j<feats.size();j++) 
                 if (j < feats.size()-1) cout << feats[j]+1 << ", "; else cout << feats[j]+1 << "],"<< endl;
             
             doTest();                  
             icrom++;
             
             cout << " }";
             
       }
       features.close();   
       cout << endl << "]" << endl;
      
    }  
    

    return 0;

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




arma::mat MC(arma::Row<size_t> labels, arma::Row<size_t> predictedLabels)
{
    // Inside the method you should call model.Predict() and compare the
    // values with the labels, in order to get the desired performance measure
    // and return it.

    arma::Row<size_t> uniqueLabels = arma::unique(labels);
    size_t numClasses = uniqueLabels.n_elem;    
    // size_t numClasses = arma::max(labels) + 1;
    arma::mat mc(numClasses, numClasses, arma::fill::zeros);
    
    // labels.print("labels: ");
    // predictedLabels.print("Plabels: ");

    for (arma::uword i = 0; i < labels.n_elem; i++)
    {
        mc( labels(i), predictedLabels(i) ) ++;            
    }

    return mc;

}


void printMC(arma::mat MC, string offset)
{
    size_t nc = MC.n_cols;
    
    string str(offset.length(), ' ');

    for (arma::uword i = 0; i < nc; i++)
    {
        if (i==0) std::cout << offset << "[[";
        else std::cout << str << " [";
        for (arma::uword j = 0; j < nc; j++)
        {
            std::cout << MC( i, j );
            if (j<(nc-1)) std::cout << ", ";
        }
        if (i<(nc-1)) std::cout << "],"<< std::endl;
        else std::cout << "]]," << std::endl;
    }
     
}    
        
     




void test(cromosoma crom, int lbits, int rank, float seed, short pobtype, double alpha, double beta, int NObjectives)
{
     string cadena, cadena1, aux;
     vector <double> aptitude;
     aptitude.resize(NObjectives);
     int Lcrom = crom.size();
     vector <int> feats_to_remove;
     
     /*double vm, rss;
     process_mem_usage(vm, rss);
     printf("-> fitsvm 1: %f\n ",rss);
     */
     
     if (Lcrom!=lbits)
     {
        cout << ">> Error en el tamaño del cromosoma <<" << endl;
        aptitude.resize(1);
        aptitude[0]=-1;
        return;
     }

     int CFeats = 0;
     for (int k=0;k<Lcrom;k++)
     {  
         if (crom[k]) CFeats++;
         else feats_to_remove.push_back(k);  // las features con valor "false" son las que voy a eliminar
     }
     
     
     if (CFeats==0) 
     {
         return;
     }    
     
     /************************************************************************************/
     // FEATURE FILTER     
     arma::uvec indices = arma::conv_to< arma::uvec >::from(feats_to_remove);       
     arma::mat TRNdataTMP = TRNdata;
     arma::mat TSTdataTMP = TSTdata;     
     TRNdataTMP.shed_rows(indices);  // las features estan en las filas !
     TSTdataTMP.shed_rows(indices);  // ELIMINO las features indicadas en el vector
     
     /************************************************************************************/
     
     cout << "  \"CLASIFICADORES\": {" << endl;
     
     for (size_t i=0;i<clasificadores.size();i++)
     {            
         arma::Row<size_t> output;
         string offset = string(25, ' ');
         tic();
        
         output = TrainTestClassifier(TRNdataTMP, TSTdataTMP, trnLabels, clasificadores[i], clasif_configs[i], optimizador, optim_configs, true);
                   
         double elapsed = toc2(); 
         
         double uar = fUAR(tstLabels,output);         
         arma::mat mc = MC(tstLabels,output);  
         string x_str;
         cout << offset << "{" << endl;
         x_str =  offset + string(5, ' ') + "\"CONFUSION_MATRIX\": ";
         printMC(mc, x_str);
         cout <<  offset << string(5, ' ') << "\"UAR\": " << uar << ","<< endl;
         cout <<  offset << string(5, ' ') << "\"ELAPSED_TIME\": "  << elapsed << endl;
         cout <<  offset << "}"; 
         if (i==(clasificadores.size()-1))
             cout << endl; 
         else 
             cout << "," << endl;     
         
         output.clear();
         mc.clear();
     }     
     cout << string(20, ' ') << "}" << endl;            

     /************************************************************************************/
    
     
     TRNdataTMP.clear();
     TSTdataTMP.clear();     
     
     return;
     
}









