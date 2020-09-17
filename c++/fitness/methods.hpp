#include <cstdlib>
#include <string>
#include <mlpack/prereqs.hpp>

#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
#include <mlpack/methods/adaboost/adaboost.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
// #include <mlpack/methods/decision_stump/decision_stump.hpp>
// #include <mlpack/core/data/one_hot_encoding.hpp>



using namespace std;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::ann;
using namespace mlpack::kmeans;



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

      recall(c) = 0.0;      
      // size_t positivePredictions = arma::sum(predictedLabels == c);
      size_t positiveLabels = arma::sum(labels == c);
      if ((positiveLabels>0) && (!(isnan(positiveLabels))) && (!(isinf(positiveLabels))))
      {    
          recall(c) = ((double) tp) / positiveLabels;
      } 

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
      recall(c) = 0.0;  
      size_t tp = arma::sum((labels == c) % (predictedLabels == c));   // el % es multiplicacion elemento a elemento en Armadillo      
      size_t positiveLabels = arma::sum(labels == c);
      if ((positiveLabels>0) && (!(isnan(positiveLabels))) && (!(isinf(positiveLabels))))
         recall(c) = ((double) tp) / positiveLabels;        
      // else cout << endl << endl << " -------------> NaN " << endl << endl;
    } 

    return arma::mean(recall);  

}




     
template <class modelo>
void EntrenarModelo (modelo &in_model, arma::mat in_data, arma::mat in_Labels, string optimizer, string optim_params) 
{
     
    if (optimizer == "sgd")
    {    
         int par3, par4;
         double par5, par6;
         bool par7;
        
         stringstream opti(optim_params);
         opti >> par3;                                            // SGD epochs        :  30
         opti >> par4;                                            // SGD batch size    :  10
         opti >> par5;                                            // SGD training speed:  0.03
         opti >> par6;                                            // SGD tolerance     :  1e-5
         opti >> par7;                                            // SGD shuffle       :  true (1/0)
         size_t numEpoches = par3;
         size_t batchSize  = par4;
         size_t numRBMIterations = in_data.n_cols * numEpoches;
         // numRBMIterations /= batchSize;                       
         ens::StandardSGD opt(par5, batchSize, numRBMIterations, par6, par7);  
         in_model.Train(in_data, in_Labels, opt);
    }
    else if (optimizer == "rmsprop")
    {    
         stringstream opti(optim_params);                      
         double par3, par5, par6, par8;
         size_t par4, par7;                                                      
         bool par9;                                                      
                   
         opti >> par3;           //   RMSProp    stepSize = 0.01, 
         opti >> par4;           //   RMSProp    batchSize = 32, 
         opti >> par5;           //   RMSProp    alpha = 0.99, 
         opti >> par6;           //   RMSProp    epsilon = 1e-8, 
         opti >> par7;           //   RMSProp    maxIterations = 100000, > EPOCHS
         opti >> par8;           //   RMSProp    tolerance = 1e-5, 
         opti >> par9;           //   RMSProp    shuffle = true)
         
         size_t numEpoches = par7;
         size_t batchSize  = par4;
         size_t numRBMIterations = in_data.n_cols * numEpoches;
         // numRBMIterations /= batchSize;   
        
         ens::RMSProp opt(par3, batchSize, par5, par6, numRBMIterations, par8, par9);
         in_model.Train(in_data, in_Labels, opt);
    }
    else if (optimizer == "adadelta")
    {
         stringstream opti(optim_params);
         double par3, par5, par6, par8;
         size_t par4, par7;                                                      
         bool par9;                                                      
         
         opti >> par3;            //   AdaDelta   stepSize = 1.0,
         opti >> par4;            //   AdaDelta   batchSize = 32,
         opti >> par5;            //   AdaDelta   rho = 0.95,
         opti >> par6;            //   AdaDelta   epsilon = 1e-6,
         opti >> par7;            //   AdaDelta   maxIterations = 100000, > EPOCHS
         opti >> par8;            //   AdaDelta   tolerance = 1e-5,
         opti >> par9;            //   AdaDelta   shuffle = true 
        
         size_t numEpoches = par7;
         size_t batchSize  = par4;
         size_t numRBMIterations = in_data.n_cols * numEpoches;
         // numRBMIterations /= batchSize;   
    
         ens::AdaDelta opt(par3, batchSize, par5, par6, numRBMIterations, par8, par9);             
         in_model.Train(in_data, in_Labels, opt);
    }
    else if (optimizer == "adagrad")
    {    
         stringstream opti(optim_params);
         double par3, par5, par7;
         size_t par4, par6;
         bool par8;
         
         opti >> par3;                      // AdaGrad       stepSize = 0.01,
         opti >> par4;                      // AdaGrad       batchSize = 32,
         opti >> par5;                      // AdaGrad       epsilon = 1e-8,
         opti >> par6;                      // AdaGrad       maxIterations = 100000, > EPOCHS
         opti >> par7;                      // AdaGrad       tolerance = 1e-5,
         opti >> par8;                      // AdaGrad       shuffle = true 
         
         size_t numEpoches = par6;
         size_t batchSize  = par4;
         size_t numRBMIterations = in_data.n_cols * numEpoches;
         // numRBMIterations /= batchSize;   
         
         ens::AdaGrad opt(par3, batchSize, par5, numRBMIterations, par7, par8);     
         in_model.Train(in_data, in_Labels, opt);

    }
    else {    
        
         stringstream opti(optim_params);                 
         double par3, par5, par6, par7, par9;                  
         size_t par4, par8;                  
         bool par10;
         
         opti >> par3;                       // Adam    stepSize = 0.001,
         opti >> par4;                       // Adam    batchSize = 32,
         opti >> par5;                       // Adam    beta1 = 0.9,
         opti >> par6;                       // Adam    beta2 = 0.999,
         opti >> par7;                       // Adam    eps = 1e-8,
         opti >> par8;                       // Adam    maxIterations = 100000, > EPOCHS
         opti >> par9;                       // Adam    tolerance = 1e-5,
         opti >> par10;                      // Adam    shuffle = true 
         
         size_t numEpoches = par8;
         size_t batchSize  = par4;
         size_t numRBMIterations = in_data.n_cols * numEpoches;
         // numRBMIterations /= batchSize;   
         
         if (optimizer == "adamax") {
            ens::AdaMax  opt(par3, batchSize, par5, par6, par7, numRBMIterations, par9, par10 );
            in_model.Train(in_data, in_Labels, opt);
         }   
         else if (optimizer == "amsgrad") {
            ens::AMSGrad opt(par3, batchSize, par5, par6, par7, numRBMIterations, par9, par10 );
            in_model.Train(in_data, in_Labels, opt);
         }   
         else if (optimizer == "nadam") {
            ens::Nadam   opt(par3, batchSize, par5, par6, par7, numRBMIterations, par9, par10 );
            in_model.Train(in_data, in_Labels, opt);
         }   
         else if (optimizer == "nadamax") {
            ens::NadaMax opt(par3, batchSize, par5, par6, par7, numRBMIterations, par9, par10 );
            in_model.Train(in_data, in_Labels, opt);
         }
         else {   
            ens::Adam opt(par3, batchSize, par5, par6, par7, numRBMIterations, par9, par10 );
            in_model.Train(in_data, in_Labels, opt);   
         }   
        
    }    
         
}        
         
       
         
        
        
arma::Row<size_t> TrainTestClassifier(arma::mat trn_data, arma::mat tst_data, arma::Row<size_t> trn_labels, string clasificador, string clasif_configs, string optimizador, string optim_configs, bool imprimir) 
{   
    
    std::transform(clasificador.begin(), clasificador.end(), clasificador.begin(), ::tolower);         
    stringstream geek(clasif_configs);  
    arma::Row<size_t> output;
    string offset = string(25, ' ');
    
    arma::Row<size_t> uniqueLabels = arma::unique(trn_labels);
    size_t numClasses = uniqueLabels.n_elem;


    if ((clasificador == "naivebayes") || (clasificador == "nb"))
    {
        if (imprimir) cout << offset << "\"NaiveBayes\":" << endl;  
        
        bool par1;
        double par2;
        geek >> par1; 
        geek >> par2; 
    
        mlpack::naive_bayes::NaiveBayesClassifier<> method (trn_data,                        //  Independent variables  
                                                            trn_labels,                      //  Dependent variables                                             
                                                            numClasses,                      //  number of classes  
                                                            par1,                            //  incrementalVariance = false,
                                                            par2 );                          //  epsilon = 1e-10 

        method.Classify(tst_data, output);  
    
    }
    else if (clasificador == "svm") 
    {
        if (imprimir) cout << offset << "\"SVM\":" << endl;   
        
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
        
        // mlpack::svm::LinearSVM<> method( trn_data, trn_labels, numClasses, par1, par2, true);          
       
        mlpack::svm::LinearSVM<> method(trn_data,                        //  Independent variables  
                                        trn_labels,                      //  Dependent variables                                             
                                        numClasses,                      //  number of classes  
                                        trn_data.n_rows,                 //  number of features
                                        par1,                            //  lambda:           L2-regularization constant.
                                        par2,                            //  delta:            Margin of difference between correct class and other classes.
                                        ens::ParallelSGD<>(par3,         // (int) maxIterations:    pSGD: Maximum number of iterations allowed (0 means no limit). (100/0)
                                                           par4,         // (int) threadShareSize:  pSGD: Number of datapoints to be processed in one iteration by each thread. (10)
                                                           par5,         // (dbl) tolerance:        pSGD: Maximum absolute tolerance to terminate the algorithm. (1e-5)
                                                           par6,         // (boo) shuffle:          pSGD: If true, the function order is shuffled; otherwise, each function is visited in linear order. (true)
                                                           par7));       // (dbl) decayPolicy:      pSGD: The step size update policy to use. (5)
        
        method.Classify(tst_data, output);  
        
    }
    else if ((clasificador == "rf") || (clasificador == "randomforest"))
    {
        if (imprimir) cout << offset << "\"RandomForest\":" << endl;  
        
        int par1, par2, par4;             
        double par3;
        geek >> par1; 
        geek >> par2; 
        geek >> par3; 
        geek >> par4; 
        mlpack::tree::RandomForest<> method( trn_data,          // Independent variables  
                                             trn_labels,        // Dependent variables
                                             numClasses,        // number of classes                                                 
                                             par1,              // numTrees = 20,
                                             par2,              // minimumLeafSize = 1,
                                             par3,              // minimumGainSplit = 1e-7,
                                             par4);             // maximumDepth = 0,
        method.Classify(tst_data, output);                                                 
    }
    else if ((clasificador == "ada") || (clasificador == "adaboost") || (clasificador == "ab"))
    {             
        if (imprimir) cout << offset <<"\"AdaBoost\":" << endl;  
        
        int par1, par2;
        double par3;
        geek >> par1; 
        geek >> par2; 
        geek >> par3;                            
        
        mlpack::perceptron::Perceptron<> weaklearner(par1);            // par1: train epochs             
        // mlpack::decision_stump::DecisionStump<> weaklearner();             
        // mlpack::tree::DecisionTree<> weaklearner();
        
        mlpack::adaboost::AdaBoost<> method( trn_data,                 // Independent variables  
                                             trn_labels,               // Dependent variables
                                             numClasses,               // number of classes
                                             weaklearner,              // weak learner
                                             par2,                     // iterations
                                             par3);                    // tolerance     
        method.Classify(tst_data, output);             
    }
    else if ((clasificador == "dt") || (clasificador == "decisiontree"))
    {
        if (imprimir) cout << offset << "\"DecisionTree\":" << endl;             
        
        int par1, par2, par3;
        geek >> par1; 
        geek >> par2; 
        geek >> par3;         
        mlpack::tree::DecisionTree<> method( trn_data,           // Independent variables
                                             trn_labels,         // Dependent variables
                                             numClasses,         // number of classes
                                             par1,               // minimum leaf size
                                             par2,               // minimum gain split 
                                             par3);              // maximum depth     
        method.Classify(tst_data, output);
    }               
    else if (clasificador == "mlp") 
    {
        if (imprimir) cout << offset << "\"MLP\":" << endl;
        
        int par1, par2;
        geek >> par1;                   // layer 1 units, if 0 -> (attribs + classes) / 2, if (-1) -> (attribs + classes) 
        geek >> par2;                   // layer 2 units, if 0 -> (attribs + classes) / 2, if (-1) -> (attribs + classes) 
        
        if (par1==0)  par1 = (trn_data.n_rows + numClasses)/2;
        if (par1==-1) par1 = (trn_data.n_rows + numClasses); 
        if (par2==0)  par2 = (trn_data.n_rows + numClasses)/2;
        if (par2==-1) par2 = (trn_data.n_rows + numClasses); 
    
        // Initialize the network.
        FFN<> model;
        model.Add<Linear<> >(trn_data.n_rows, par1);
        model.Add<SigmoidLayer<> >();
        model.Add<Linear<> >(par1, par2);
        model.Add<SigmoidLayer<> >();
        model.Add<Linear<> >(par2, numClasses);
        model.Add<LogSoftMax<> >();             

        // Train the model.
        arma::mat trn_labels_Mat, pred_one_hot;
        trn_labels_Mat = arma::conv_to<arma::mat>::from(trn_labels+1);                                
        EntrenarModelo<FFN<>> (model, trn_data, trn_labels_Mat, optimizador, optim_configs);             
        model.Predict(tst_data, pred_one_hot);                          
        output.zeros(pred_one_hot.n_cols);
        // Find index of max prediction for each data point and store in "prediction"
        for (size_t p = 0; p < pred_one_hot.n_cols; ++p)
        {
           output(p) = arma::as_scalar(arma::find(arma::max(pred_one_hot.col(p)) == pred_one_hot.col(p), 1));                 
        }                              
        pred_one_hot.clear();

    }         
    else if (clasificador == "rbf") 
    {
        if (imprimir) cout << offset << "\"RBF\":" << endl;

        int par1;
        double par2;
        geek >> par1;                   // number of layer 1 units / centroids
        geek >> par2;                   // betas: The beta value to be used with centres (double, 0).
                
        arma::mat centroids;
        KMeans<> kmeans;
        kmeans.Cluster(trn_data, par1, centroids);                     // centres: The centres calculated using k-means of data (arma::mat).         
        
        // Initialize the network.
        FFN<> model;
        model.Add<RBF<> >(trn_data.n_rows, par1, centroids, par2);     // inSize: The number of input units (size_t).
                                                                       // outSize: The number of output units (size_t).                                                                            
        model.Add<Linear<> >(par1, numClasses);
        model.Add<LogSoftMax<> >();             
        
        // Train the model.
        arma::mat trn_labels_Mat, pred_one_hot;
        trn_labels_Mat = arma::conv_to<arma::mat>::from(trn_labels+1);         
        EntrenarModelo<FFN<>> (model, trn_data, trn_labels_Mat, optimizador, optim_configs);             
        model.Predict(tst_data, pred_one_hot);                          
        output.zeros(pred_one_hot.n_cols);
        // Find index of max prediction for each data point and store in "prediction"
        for (size_t p = 0; p < pred_one_hot.n_cols; ++p)
        {
           output(p) = arma::as_scalar(arma::find(arma::max(pred_one_hot.col(p)) == pred_one_hot.col(p), 1));                 
        }                              
        pred_one_hot.clear();

    }
    
    return output;
        
}
