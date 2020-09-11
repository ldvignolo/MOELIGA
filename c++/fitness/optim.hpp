#include <cstdlib>
#include <string>
#include <mlpack/prereqs.hpp>

using namespace std;
using namespace mlpack;
using namespace data;


     
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
         numRBMIterations /= batchSize;                       
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
         opti >> par7;           //   RMSProp    maxIterations = 100000, 
         opti >> par8;           //   RMSProp    tolerance = 1e-5, 
         opti >> par9;           //   RMSProp    shuffle = true)
        
         ens::RMSProp opt(par3, par4, par5, par6, par7, par8, par9);
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
        opti >> par7;            //   AdaDelta   maxIterations = 100000,
        opti >> par8;            //   AdaDelta   tolerance = 1e-5,
        opti >> par9;            //   AdaDelta   shuffle = true 
    
        ens::AdaDelta opt(par3, par4, par5, par6, par7, par8, par9);             
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
        opti >> par6;                      // AdaGrad       maxIterations = 100000,
        opti >> par7;                      // AdaGrad       tolerance = 1e-5,
        opti >> par8;                      // AdaGrad       shuffle = true 
    
        ens::AdaGrad opt(par3, par4, par5, par6, par7, par8);     
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
        opti >> par8;                       // Adam    maxIterations = 100000,
        opti >> par9;                       // Adam    tolerance = 1e-5,
        opti >> par10;                      // Adam    shuffle = true 

        if (optimizer == "adamax") {
           ens::AdaMax  opt(par3, par4, par5, par6, par7, par8, par9, par10 );
           in_model.Train(in_data, in_Labels, opt);
        }   
        else if (optimizer == "amsgrad") {
           ens::AMSGrad opt(par3, par4, par5, par6, par7, par8, par9, par10 );
           in_model.Train(in_data, in_Labels, opt);
        }   
        else if (optimizer == "nadam") {
           ens::Nadam   opt(par3, par4, par5, par6, par7, par8, par9, par10 );
           in_model.Train(in_data, in_Labels, opt);
        }   
        else if (optimizer == "nadamax") {
           ens::NadaMax opt(par3, par4, par5, par6, par7, par8, par9, par10 );
           in_model.Train(in_data, in_Labels, opt);
        }
        else {   
           ens::Adam opt(par3, par4, par5, par6, par7, par8, par9, par10 );
           in_model.Train(in_data, in_Labels, opt);   
        }   
        
    } 

}

