//----------------------------------------------------------------------------
//
//  Funcion de Fitness 
//
//
// 
//  17/06/20 -> Classifiers SVM and LVM + Relief Measure
// 
//  LDV
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
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
#include <mlpack/methods/adaboost/adaboost.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/scaler_methods/mean_normalization.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
// #include <mlpack/methods/decision_stump/decision_stump.hpp>
// #include <mlpack/core/data/one_hot_encoding.hpp>
// #include <mlpack/core/cv/metrics/accuracy.hpp>

#include "loadarff.hpp"
#include "optim.hpp"

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



using namespace std;

vector <string> clasificadores;
vector <string> clasif_configs;
string optimizador, optim_configs;
size_t  numClasses;

arma::mat TRNdata;
arma::Row<size_t> trnLabels;


vector <double> fitness(cromosoma crom, int lbits, int rank, float seed, short pobtype, int NObjectives, unsigned ntest, float validationSize, bool c_eval);
double test(string configs, struct svm_problem datos, struct svm_model *modelo, vector <int> labels);
void process_mem_usage(double& vm_usage, double& resident_set);
double distL1(arma::mat x, arma::mat y);
double Rmeasure(arma::mat data, arma::Row<size_t> labels, float prop);
double fUAR(arma::Row<size_t> labels, arma::Row<size_t> predictedLabels);


/*------------------------------------------------------------------------------------------------*/


class UAR
{
  public:  
  //
  // This evaluates the metric given a trained model and a set of data (with
  // labels or responses) to evaluate on.  The data parameter will be a type of
  // Armadillo matrix, and the labels will be the labels that go with the model.
  //
  // If you know that your model is a classification model (and thus that
  // ResponsesType will be arma::Row<size_t>), it is ok to replace the
  // ResponsesType template parameter with arma::Row<size_t>.
  //
  template<typename MLAlgorithm, typename DataType, typename ResponsesType>
  static double Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const ResponsesType& labels)
  {
    // Inside the method you should call model.Predict() and compare the
    // values with the labels, in order to get the desired performance measure
    // and return it.
      
    arma::Row<size_t> predictedLabels;
    model.Classify(data, predictedLabels);

    
    arma::Row<size_t> uniqueLabels = arma::unique(labels);
    size_t numClasses = uniqueLabels.n_elem;      
    // size_t numClasses = arma::max(labels) + 1;

    arma::vec recall = arma::vec(numClasses);
    for (size_t c = 0; c < numClasses; ++c)
    {
      size_t tp = arma::sum((labels == c) % (predictedLabels == c));   // el % es multiplicacion elemento a elemento en Armadillo
      // size_t positivePredictions = arma::sum(predictedLabels == c);
      size_t positiveLabels = arma::sum(labels == c);
      recall(c) = double(tp) / positiveLabels;
    }

    return arma::mean(recall);  

  }
  
};


double fUAR(arma::Row<size_t> labels, arma::Row<size_t> predictedLabels)
{
    // Inside the method you should call model.Predict() and compare the
    // values with the labels, in order to get the desired performance measure
    // and return it.

    arma::Row<size_t> uniqueLabels = arma::unique(labels);
    size_t numClasses = uniqueLabels.n_elem;    
    // size_t numClasses = arma::max(labels) + 1;

    arma::vec recall = arma::vec(numClasses);
    for (size_t c = 0; c < numClasses; ++c)
    {
      size_t tp = arma::sum((labels == c) % (predictedLabels == c));   // el % es multiplicacion elemento a elemento en Armadillo
      // size_t positivePredictions = arma::sum(predictedLabels == c);
      size_t positiveLabels = arma::sum(labels == c);
      recall(c) = double(tp) / positiveLabels;
    }

    return arma::mean(recall);  

}

/*------------------------------------------------------------------------------------------------*/



double sigmoid(double value, double lambda, double gamma)
{
    return  1.0 / (1.0 + exp( (-1.0)*(value+gamma)*lambda ) );
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
    
    string trnfile = SETTINGS.get_str("trnfile");
    optimizador = SETTINGS.get_str("Optimizer");  
    string clasificador = SETTINGS.get_str("classifier");    
    clasificadores = SplitWords(clasificador);
    string aux;
    
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
    // we need to extract the labels from the last dimension of the dataset and remove the labels from the training set:
    trnLabels =  arma::conv_to<arma::Row<size_t>>::from(TRNdata.row(TRNdata.n_rows - 1));
    TRNdata.shed_row(TRNdata.n_rows - 1); // elimino fila correspondiente a las etiquetas    
    arma::Row<size_t> uniqueLabels = arma::unique(trnLabels);
    numClasses = uniqueLabels.n_elem;
    arma::mat data_aux;
    
    bool normalizar = SETTINGS.get_bool("Normalizar");        
    bool estandarizar = SETTINGS.get_bool("Estandarizar");
        
    if (normalizar)
    {         
       // Fit the features.
       data::MeanNormalization scale;       
       scale.Fit(TRNdata);
       // Scale the features.
       scale.Transform(TRNdata, data_aux);
       TRNdata = data_aux;  
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
    }            
    
    bool Obj2Sigmod = SETTINGS.get_bool("Obj2Sigmod"); // false por omision     
    float SigmLambda = SETTINGS.get_dbl("SigmLambda"); 
    float SigmGamma = SETTINGS.get_dbl("SigmGamma");
    if (SigmLambda>500.0) SigmLambda = 1.5; // valor por omision
    if (SigmGamma>500.0) SigmGamma = 0.0; // valor por omision
    
    
    float validationSize = SETTINGS.get_dbl("validationSize");
    unsigned ntest = SETTINGS.get_int("NTests");
    if (ntest<=0) ntest=1;       
    
    if ( (validationSize<=0) || (validationSize>1) ) 
    {        
        // compatibilidad con settings viejos
        float ptrain = SETTINGS.get_dbl("ptrain");     
        if ( (ptrain<=0) || (ptrain>100) ) ptrain = 70;        
        validationSize = ((float) ptrain) /100;    
    }

    /*--------------------------------*/

    srand ( unsigned ( std::time(0) ) );
    
    /*--------------------------------------------------*/
    
    // MPI_Recv(params, 2, MPI_INTEGER, 0, id, parent_comm, &status);
    MPI_Recv(&seed, 1, MPI_FLOAT, 0, id, parent_comm, &status);
    MPI_Recv(&NObjectives, 1, MPI_INTEGER, 0, id, parent_comm, &status);     
    
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
        
           aptitud = fitness(cromovect, lcrom, rank, seed, pobtype, NObjectives, ntest, validationSize, c_eval); 
           
           if ((Obj2Sigmod) && (1<NObjectives))
               aptitud[1] = sigmoid(aptitud[1], SigmLambda, SigmGamma); 
           
           for (i=0;i<NObjectives;i++) fit[i] = aptitud[i];
           MPI_Send(fit, NObjectives, MPI_DOUBLE, 0, id, parent_comm);
        }
        
    }
    
    free(cromo);
    free(fit);
    
    MPI_Finalize();
    
    return 0;

}



double distL1(arma::mat x, arma::mat y)
{
    double dist = 0.0;
    for (arma::uword i=0;i<x.n_rows;i++)
        dist = dist + fabs(x(i)-y(i));
    dist = dist / x.n_rows;
    
    return dist;
}




double Rmeasure(arma::mat data, arma::Row<size_t> labels, float prop)
{

    unsigned int Nd = data.n_cols;    
    unsigned int Nr = ceil(prop*Nd);
    
    vector <unsigned int> indI;
    
    indI.resize(Nd);
    for (unsigned i=0;i<Nd;i++) indI[i]=i;
    random_shuffle ( indI.begin(), indI.end() );
    
    double nmiss, nhit, dist, measure = 0.0;
    for (unsigned k=0;k<Nr;k++){
        
        nmiss = DBL_MAX;
        nhit = DBL_MAX;        
        for (unsigned j=0;j<Nd;j++){
            if (j!=indI[k]) 
            {                
                dist = distL1(data.col(j), data.col(indI[k]));
                if (labels(j) == labels(indI[k])) {
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






vector <double> fitness(cromosoma crom, int lbits, int rank, float seed, short pobtype, int NObjectives, unsigned ntest, float validationSize, bool c_eval)
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
        return aptitude;
     }

     int CFeats = 0;
     for (int k=0;k<Lcrom;k++)
     {  
         if (crom[k])  CFeats++;
         else feats_to_remove.push_back(k);  // las features con valor "false" son las que voy a eliminar
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
         
     /************************************************************************************/
     // FEATURE FILTER     
     arma::uvec indices = arma::conv_to< arma::uvec >::from(feats_to_remove);       
     arma::mat TRNdataTMP = TRNdata; // las features estan en las filas !
     TRNdataTMP.shed_rows(indices);  // ELIMINO las features indicadas en el vector     
     arma::Row<size_t> trnLabelsTMP = trnLabels;
     size_t NData = TRNdataTMP.n_cols;
     /************************************************************************************/
    
     double cUAR = 0.0, fit_aux = 0.0, mR = 0.0;
     
     for (unsigned jk=0;jk<ntest;jk++)
     {            
         // TRAIN DATA SHUFFLE      
         arma::Row<size_t> auxLabels;    
         arma::mat data_aux;
         math::ShuffleData(TRNdataTMP, trnLabelsTMP, data_aux, auxLabels);
         TRNdataTMP = data_aux;  
         trnLabelsTMP = auxLabels;
         data_aux.clear();                   
         auxLabels.clear();             
         
         if (c_eval) 
         {            

             for (size_t i=0;i<clasificadores.size();i++)
             {                    
                 string clasificador = clasificadores[i];
                 std::transform(clasificador.begin(), clasificador.end(), clasificador.begin(), ::tolower);         
                 stringstream geek(clasif_configs[i]);          

                
                 if ((clasificador == "naivebayes") || (clasificador == "nb"))
                 {
                     bool par1;
                     double par2;
                     geek >> par1; 
                     geek >> par2; 
                    
                     cv::SimpleCV<mlpack::naive_bayes::NaiveBayesClassifier<>, UAR> val(validationSize, TRNdataTMP, trnLabelsTMP, numClasses); 
                     
                     cUAR = val.Evaluate( par1,                            //  incrementalVariance = false,
                                          par2 );                          //  epsilon = 1e-10 
                      
                 }    
                 else if (clasificador == "svm") 
                 {
                     double par1, par2, par5, par7;
                     int par3, par4; 
                     bool par6;
                     geek >> par1; 
                     geek >> par2; 
                     geek >> par3;              
                     geek >> par4; 
                     geek >> par5; 
                     geek >> par6;              
                     geek >> par7;        
                     
                     cv::SimpleCV<mlpack::svm::LinearSVM<>, UAR> val(validationSize, TRNdataTMP, trnLabelsTMP, numClasses);                     

                     cUAR = val.Evaluate( TRNdataTMP.n_rows,
                                          par1,                             //  lambda:           L2-regularization constant.
                                          par2,                             //  delta:            Margin of difference between correct class and other classes.
                                          ens::ParallelSGD<>( par3,         //  maxIterations:    pSGD: Maximum number of iterations allowed (0 means no limit). (100/0)
                                                              par4,         //  threadShareSize:  pSGD: Number of datapoints to be processed in one iteration by each thread. (10)
                                                              par5,         //  tolerance:        pSGD: Maximum absolute tolerance to terminate the algorithm. (1e-5)
                                                              par6,         //  shuffle:          pSGD: If true, the function order is shuffled; otherwise, each function is visited in linear order. (true)
                                                              par7 ));      //  decayPolicy:      pSGD: The step size update policy to use. (5)                                          
                     
                     // cv::SimpleCV<mlpack::svm::LinearSVM<>, UAR> cv3(validationSize, TRNdataTMP, trnLabelsTMP, numClasses);  
                     // cUAR = cv3.Evaluate(0.0001 /*lambda*/, 1.0 /*delta*/, false /*shuffle*/); 
                     
                 }  
                 else if (clasificador == "rf") 
                 {
                     int par1, par2, par4;             
                     double par3;
                     geek >> par1; 
                     geek >> par2; 
                     geek >> par3; 
                     geek >> par4; 

                     cv::SimpleCV<mlpack::tree::RandomForest<>, UAR> val(validationSize, TRNdataTMP, trnLabelsTMP, numClasses);
                     
                     cUAR = val.Evaluate( par1,              // numTrees = 20,
                                          par2,              // minimumLeafSize = 1,
                                          par3,              // minimumGainSplit = 1e-7,
                                          par4 );            // maximumDepth = 0,
                     
                 }               
                 else if (clasificador == "ada") 
                 {
                     int par1, par2;
                     double par3;
                     geek >> par1; 
                     geek >> par2; 
                     geek >> par3;  
                     
                     cv::SimpleCV<mlpack::adaboost::AdaBoost<>, UAR> val(validationSize, TRNdataTMP, trnLabelsTMP, numClasses);
                     
                     mlpack::perceptron::Perceptron<> weaklearner(par1);        // par1: train epochs             
                     
                     cUAR = val.Evaluate( weaklearner,                        // weak learner
                                          par2,                               // iterations
                                          par3);                              // tolerance     
                     
                 }               
                 else if (clasificador == "dt") 
                 {
                     int par1, par2, par3;
                     geek >> par1; 
                     geek >> par2; 
                     geek >> par3;         
                     
                     cv::SimpleCV<mlpack::tree::DecisionTree<>, UAR> val(validationSize, TRNdataTMP, trnLabelsTMP, numClasses);
                     
                     cUAR = val.Evaluate( par1,               // minimum leaf size
                                          par2,               // minimum gain split
                                          par3);              // maximum depth     
                     
                 }               
                 else if (clasificador == "mlp")    
                 {     
                     int par1, par2;
                     geek >> par1;                   // layer 1 units, if 0 -> (attribs + classes) / 2, if (-1) -> (attribs + classes
                     geek >> par2;                   // layer 2 units, if 0 -> (attribs + classes) / 2, if (-1) -> (attribs + classes
                
                     /*
                     int par3, par4;
                     double par5, par6;
                     bool par7;
                     geek >> par3;                   // SGD epochs        :  30
                     geek >> par4;                   // SGD batch size    :  10
                     geek >> par5;                   // SGD training speed:  0.03
                     geek >> par6;                   // SGD tolerance     :  1e-5
                     geek >> par7;                   // SGD shuffle       :  true (1/0)
                     */
                     
                     if (par1==0)  par1 = (TRNdataTMP.n_rows + numClasses)/2;
                     if (par1==-1) par1 = (TRNdataTMP.n_rows + numClasses); 
                     if (par2==0)  par2 = (TRNdataTMP.n_rows + numClasses)/2;
                     if (par2==-1) par2 = (TRNdataTMP.n_rows + numClasses); 
                     
                     // Initialize the network.
                     FFN<> model;
                     model.Add<Linear<> >(TRNdataTMP.n_rows, par1);
                     model.Add<SigmoidLayer<> >();
                     model.Add<Linear<> >(par1, par2);
                     model.Add<SigmoidLayer<> >();
                     model.Add<Linear<> >(par2, numClasses);
                     model.Add<LogSoftMax<> >();             
                     
                     /*
                     size_t numEpoches = par3;
                     size_t batchSize  = par4;                     
                     size_t numRBMIterations = NData * numEpoches;
                     numRBMIterations /= batchSize;
                     ens::StandardSGD opt(par5, batchSize, numRBMIterations, par6, par7);             
                     */
                     
                     // Cross Validation
                     arma::uvec trnIdx = arma::regspace<arma::uvec>(0, 1, floor(1.0-validationSize)*(NData-1));  
                     arma::uvec tstIdx = arma::regspace<arma::uvec>(0, 1, ceil(validationSize*(NData-1)));  

                     // Train the model.                      
                     arma::Row<size_t> output;                     
                     arma::mat trnLabelsMat, pred_one_hot;
                     trnLabelsMat = arma::conv_to<arma::mat>::from(trnLabelsTMP+1);      
                     model.ResetParameters();                     
                     // model.Train(TRNdataTMP.cols(trnIdx), trnLabelsMat.cols(trnIdx), opt);     
                     EntrenarModelo<FFN<>> (model, TRNdataTMP.cols(trnIdx), trnLabelsMat.cols(trnIdx), optimizador, optim_configs);                      
                     model.Predict(TRNdataTMP.cols(tstIdx), pred_one_hot);                         
                     output.zeros(pred_one_hot.n_cols);
                     // Find index of max prediction for each data point and store in "prediction"
                     for (size_t p = 0; p < pred_one_hot.n_cols; ++p)
                     {
                        output(p) = arma::as_scalar(arma::find(arma::max(pred_one_hot.col(p)) == pred_one_hot.col(p), 1));                 
                     }                              
                     pred_one_hot.clear();
                     cUAR = fUAR(trnLabelsTMP.cols(tstIdx),output);
                     
                 }                       
                 else if (clasificador == "rbf")
                 {                     
                     
                     int par1;
                     double par2;
                     geek >> par1;                   // number of layer 1 units / centroids
                     geek >> par2;                   // betas: The beta value to be used with centres (double, 0).
  
                     /*
                     int par1, par3, par4;
                     double par2, par5, par6;
                     bool par7;
                     geek >> par1;                   // number of layer 1 units / centroids
                     geek >> par2;                   // betas: The beta value to be used with centres (double, 0).
                     geek >> par3;                   // SGD epochs        :  30
                     geek >> par4;                   // SGD batch size    :  10
                     geek >> par5;                   // SGD training speed:  0.03
                     geek >> par6;                   // SGD tolerance     :  1e-5
                     geek >> par7;                   // SGD shuffle       :  true (1/0)
                     */
                     
                     arma::mat centroids;
                     KMeans<> kmeans;
                     kmeans.Cluster(TRNdataTMP, par1, centroids);                   // centres: The centres calculated using k-means of data (arma::mat).         
                     
                     // Initialize the network.
                     FFN<> model;
                     model.Add<RBF<> >(TRNdataTMP.n_rows, par1, centroids, par2);   // inSize: The number of input units (size_t).
                                                                                    // outSize: The number of output units (size_t).                                                                            
                     model.Add<Linear<> >(par1, numClasses);
                     model.Add<LogSoftMax<> >();             
                     
                     /*
                     size_t numEpoches = par3;
                     size_t batchSize  = par4;
                     size_t numRBMIterations = TRNdataTMP.n_cols * numEpoches;
                     numRBMIterations /= batchSize;
                     ens::StandardSGD opt(par5, batchSize, numRBMIterations, par6, par7);             
                     */
                     
                     // Cross Validation
                     arma::uvec trnIdx = arma::regspace<arma::uvec>(0, 1, floor(1.0-validationSize)*(NData-1));  
                     arma::uvec tstIdx = arma::regspace<arma::uvec>(0, 1, ceil(validationSize*(NData-1)));                                            
                     
                     // Train the model.
                     arma::Row<size_t> output;                                                               
                     arma::mat trnLabelsMat, pred_one_hot;                     
                     trnLabelsMat = arma::conv_to<arma::mat>::from(trnLabelsTMP+1);
                     // model.Train(TRNdataTMP.cols(trnIdx), trnLabelsMat.cols(trnIdx), opt);  
                     EntrenarModelo<FFN<>> (model, TRNdataTMP.cols(trnIdx), trnLabelsMat.cols(trnIdx), optimizador, optim_configs);                      
                     model.Predict(TRNdataTMP.cols(tstIdx), pred_one_hot);                                            
                     output.zeros(pred_one_hot.n_cols);
                     // Find index of max prediction for each data point and store in "prediction"
                     for (size_t p = 0; p < pred_one_hot.n_cols; ++p)
                     {
                        output(p) = arma::as_scalar(arma::find(arma::max(pred_one_hot.col(p)) == pred_one_hot.col(p), 1));                 
                     }                              
                     pred_one_hot.clear();                     
                     cUAR = fUAR(trnLabelsTMP.cols(tstIdx),output);                     
                     
                 }                      
                 fit_aux = fit_aux + cUAR;                 
             }
             fit_aux = fit_aux / clasificadores.size();
            
         } else fit_aux = 0.0;
         
         if (NObjectives > 2) 
         {
             mR = mR + Rmeasure(TRNdataTMP, trnLabelsTMP, validationSize);                     
         }
     
     }     
     fit_aux = fit_aux / ntest;
     
     aptitude[0] = fit_aux;
     
     if (NObjectives > 2)
         aptitude[2] = mR/ntest;
       
     if (Lcrom!=0) 
     {
        aptitude[1] = (double (Lcrom-CFeats))/Lcrom;        
     } 
     else aptitude[1] = 0;
     
     return aptitude;
     
}


