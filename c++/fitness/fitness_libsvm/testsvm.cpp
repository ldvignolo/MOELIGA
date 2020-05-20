//----------------------------------------------------------------------------
//
//  Función de Fitness para la versión paralela
//
//  Este programa es llamado desde ag_lin_p, como
//  ejecutable a parte para correr en cada nodo (MPI_COMM_spawn).
//
//
//  Esta version es para utilizar con los datos ASM de caras (proyecto brasil)
//  Miércoles 21 Marzo 2012
//
// 
//  USAGE: ./testsvm cromo_GCM.txt cfg gcm_SETTINGS_Test.cfg
// 
//  Leandr0
//
//----------------------------------------------------------------------------


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

#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR


#include "../../GA/types.h"
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

//--------------------------------------

#include <unistd.h>
#include <ios>

#include <stack>
#include <ctime>


using namespace std;


struct svm_problem trnD,tstD;
struct svm_node *trn_space, *tst_space;

vector <string> str_labels;

const char* trnfile="";
const char* tstfile="";

string configs_trn = "";
string configs_tst = "";
string salida = "";

vector <double> fitness(cromosoma crom, int lbits, int rank, float seed, short pobtype, double alpha, double beta, int NObjectives);
vector <string> SplitWords(string strString);
struct svm_model* train(unsigned Nfeat, struct svm_problem datos);
double test(string configs, struct svm_problem datos, struct svm_model *modelo, vector <int> labels);
void process_mem_usage(double& vm_usage, double& resident_set);
void scale_data(unsigned Ncols);



/*================================
  FUNCIONES Y DEFINICIONES UTILES
  ================================*/
stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << endl << endl;
    tictoc_stack.pop();
}


int main(int argc, char** argv)
{
    cromosoma cromovect;
    vector <double> aptitud;
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
    
//     int* cromo = (int*) malloc(lcrom * sizeof(int));
    
    //==================================
    // LEVANTO ARCHIVO DE CONFIGURACION
    //----------------------------------
    
    Dictionary SETTINGS(cfg_settings.c_str());
          
    double alpha = SETTINGS.get_dbl("alpha");  
    double beta  = SETTINGS.get_dbl("beta");
            
    string aux1 = "", aux2 = "";
    aux1 = SETTINGS.get_str("TESTFINALtrnfile");
    aux2 = SETTINGS.get_str("TESTFINALtstfile");
    
    if (aux1.compare("None") == 0){
        aux1 = SETTINGS.get_str("trnfile");
    }     
    if (aux2.compare("None") == 0){
        aux2 = SETTINGS.get_str("tstfile");
    } 
    
    trnfile=aux1.c_str();
    tstfile=aux2.c_str();
    
      
	configs_trn = SETTINGS.get_str("configs_trn");
	configs_tst = SETTINGS.get_str("configs_tst");
	salida = SETTINGS.get_str("svm_output");        
    
    configs_trn.insert(configs_trn.begin(),' ');
    configs_trn.insert(configs_trn.end(),' ');
    configs_tst.insert(configs_tst.begin(),' ');
    configs_tst.insert(configs_tst.end(),' ');      
    
    
	// configs_trn=" -s 0 -t 0 -q ";
	// configs_tst=" -q ";                
	// cout << "-> " << configs_trn << endl;
	// cout << "-> " << configs_tst << endl;
	
	/*--------------------------------*/
	
	unsigned Nfeat;
	ReadProblemArff(trnfile, Nfeat, trnD, &trn_space[0]);
	ReadProblemArff(tstfile, Nfeat, tstD, &tst_space[0], str_labels);

	bool normalizar = SETTINGS.get_bool("Normalizar");        
	
	if (normalizar){
          scale_data(Nfeat);
	}  
	
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
	
	/*--------------------------------*/
	
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

        tic();        
       
        aptitud = fitness(cromovect, lcrom, 0, seed, pobtype, alpha, beta, NObjectives); 

        if (salida.compare("labels") != 0){
           
           cout << "> Features: "<< feats.size() << endl;
           
        }
        
        cout << endl;          
        toc();
        cout << "============================================="<< endl<< endl;                
        // cout << "-> todo leido"<< endl;
        // lcrom=Nfeat;
        // for (i=0;i<lcrom;i++) cromovect.push_back(true);
        // aptitud = fitness(cromovect, lcrom, rank, seed, pobtype); 
	    // for (unsigned r=0;r<feats.size();r++) cout << feats[r]<< " " ; cout << endl;
        
      //  free(cromo); lcrom=0;
    };
    
    
    
	int icrom=1;
	if (argc < 2){
	  
	  for (unsigned j=0;j<Nfeat;j++) feats.push_back(j);
      doTest();
	  
	} else
	{
	    // filename = argv[1];
	    ifstream features (filename.c_str());
      
        /*
	    if (features.is_open())
	    {

	      while (features >> aux_int)
	      {
		  aux_int = aux_int-1;  // en la lista que imprime agp el primer elemento es 1, no 0
		  feats.push_back(aux_int);
	      }
	      features.close();
	      
	    } else {
	      
	      for (unsigned j=0;j<Nfeat;j++) feats.push_back(j);
	      
	    }  
	    */
	    
	    if(!features) {
            cout<<"Couldn't open the file"<<endl;
            exit(1);
        }
        
        string line;           
        // considero que puede haber varios cromosomas en un mismo archivo, uno por linea, entonces hago el test para cada linea
        while ( getline( features, line )) {
        
             feats.resize(0);
             stringstream str(line);
             
             cout << "Cromo Nr: " << icrom << endl << endl; 
             
             while (str >> aux_int) {
                 aux_int = aux_int-1;
                 feats.push_back(aux_int);
                 // cout << aux_int << " ";
             }             
             // cout << endl << endl; 
             doTest();                  
             icrom++;
         }
	     features.close();   
	  
	}  
	
	free(Xargv);
	
	/*--------------------------------*/

    
// 	free(Xargv);
	free(trnD.y);
	free(tstD.y);
	free(tst_space);
	free(trn_space);
	

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












vector <double> fitness(cromosoma crom, int lbits, int rank, float seed, short pobtype, double alpha, double beta, int NObjectives)
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
     
     /************************************************************************************/
     
     size_t elements;
     struct svm_model *modelo;
     struct svm_problem trnD_aux, tstD_aux;
     struct svm_node *trn_space_aux, *tst_space_aux;
     
     /* -------------------------------------------------*/
     
     int idx, Ncols = CFeats+1;
     
     trnD_aux.l = trnD.l;              // number of rows
     elements = Ncols * trnD.l;        // total number of features (nfeat * prob.l)     
     trnD_aux.y  = Malloc(double,trnD_aux.l);
     trnD_aux.x  = Malloc(struct svm_node *,trnD_aux.l);
     trn_space_aux = Malloc(struct svm_node,elements);

     tstD_aux.l = tstD.l;              // number of rows
     elements = Ncols * tstD.l;        // total number of features (nfeat * prob.l)     
     tstD_aux.y  = Malloc(double,tstD_aux.l);
     tstD_aux.x  = Malloc(struct svm_node *,tstD_aux.l);
     tst_space_aux = Malloc(struct svm_node,elements);          
     
     /* -------------------------------------------------*/
     
     int j=0;
     for (int i=0;i<trnD.l;i++)
     {
        trnD_aux.x[i] = &trn_space_aux[j];           
        trnD_aux.y[i] = trnD.y[i];
        idx = 1;
        for (int k=0;k<Lcrom;k++)
        {   
           if (crom[k]){
              trn_space_aux[j].index = idx;
              trn_space_aux[j].value = trnD.x[i][k].value;
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
 
     
     modelo = train(CFeats, trnD_aux);
     // cout << "--> " << rank << " : " << "entrenado" << endl;
     aptitude[1] = test(configs_tst, tstD_aux, modelo, tst_labels);
     svm_free_and_destroy_model(&modelo); 

     // cout << "--> " << rank << " : " << "testeado" << endl;
    
     /************************************************************************************/

     free(trnD_aux.y);
     free(trn_space_aux);
     free(tstD_aux.y);
     free(tst_space_aux);
        
     if (Lcrom!=0)
      aptitude[2] = (double (Lcrom-CFeats))/Lcrom;
     else 
      aptitude[2] = 0;
     
     aptitude[0] =  alpha*aptitude[1] + beta*aptitude[2];
     
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
	
	double rr, UAR=0;
	  
	if (salida.compare("labels") != 0){
	
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

	    UAR=0;
	    for (i=0;i<nclass;i++){
	      rr = 0;
	      for (k=0;k<nclass;k++)
		  rr = rr + MC[ i ][ k ];
	      UAR = UAR + MC[ i ][ i ]/rr;
	    }
	    UAR = UAR/nclass;
	    
	    for (i=0;i<nclass;i++){
	      for (k=0;k<nclass;k++)
		  cout << MC[ i ][ k ] << " ";
		cout << "\n"; }
	    cout << "> UAR: " << UAR << endl;  
	
	} else {  
	  
	    for (i=0;i<(int)pred_labels.size();i++) cout << str_labels[pred_labels[i]-1] << endl;
	  
	}  

	return(UAR);
}

