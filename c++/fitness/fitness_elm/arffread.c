#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <iterator> 
#include <list> 
#include <map>
#include <string.h>
#include "../libsvm-3.20/svm.h"
// #include "svm-train.c"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;

typedef vector<string> Tcadena;

// void ReadProblemArff(const char *filename);

// void GetProblemParams(const char *filename, unsigned &Nrows, unsigned &Nfeat, unsigned &Nlabels, unsigned &lbl_idx);

// struct svm_problem prob;
// struct svm_node *x_space;


void chomp( string &s)
{
  
  s.erase(s.find_last_not_of(" \n\r\t")+1);

}



void GetProblemParams(const char *filename, unsigned &Nrows, unsigned &Nfeat, unsigned &Nlabels, unsigned &lbl_idx, unsigned &Nlines)
{
     string cadena, tmp, label;
     unsigned i=0;
     int cmp1, cmp2, cmp3, cmp4, cmp5;
     bool flag1=false, flag2=false;
     
     ifstream datos;
     datos.open(filename, ifstream::in);
     
     Nfeat = 0;
     Nlabels = 0;
     Nrows = 0;
     Nlines = 0;
     
     bool salirEOF=false;
     while (!salirEOF) { // cuento las filas del archivo
        if (getline(datos,cadena)) i++; else salirEOF=true;
        
	cmp1 = (int) cadena.find("@attribute");
        cmp2 = (int) cadena.find("@ATTRIBUTE");
	cmp3 = (int) cadena.find("@Attribute");
	cmp4 = (int) cadena.find("{");
	cmp5 = (int) cadena.find("}");
	
	if (((cmp1>=0)||(cmp2>=0)||(cmp3>=0))&&((cmp4<0)||(cmp5<0))) 
	{
	      Nfeat++;
	      
	} else if ((cmp4>=0)&&(cmp5>=0)) {
	  
	      flag1 = true;
	      cmp4=cmp4+1;
	      cmp5=cmp5-1;
	      tmp = cadena.substr(cmp4,cmp5-cmp4+1);	      
	      cmp4 = 1;
	      Nlabels = 1;
	      lbl_idx = Nfeat;
              
	      while (cmp4>0) 
	      {        
		  cmp4 = (int) tmp.find(",");
		  if (cmp4>0) Nlabels++;
		  label = tmp.substr(0,cmp1);
		  tmp = tmp.substr(cmp4+1,tmp.length()-cmp1);       
	      }
	}
	
	cmp1 = (int) cadena.find("@DATA");
        cmp2 = (int) cadena.find("@data");
	cmp3 = (int) cadena.find("@Data");
	
	if ((cmp1>=0)||(cmp2>=0)||(cmp3>=0)) {
	   flag2 = true;
	   if (getline(datos,cadena)) i++; else salirEOF=true;
	}   
	
	if (flag1&&flag2) {
	  
	    cmp1 = (int) cadena.find(",");
	    if (cmp1>=0) Nrows++;
	}
	
     }     
     
     Nlines = i;
     datos.close();
     
}

// struct svm_problem prob;
// struct svm_node *x_space;






void ReadProblemArff(const char *filename, unsigned &Nfeat, struct svm_problem &prob, struct svm_node *x_space)
{
     string cadena, tmp, label;
     unsigned jj, j=0;
     int i=0;
     double value;
     ifstream datos;
     size_t elements;
     unsigned Nrows, Nlabels, lbl_idx, Nlines, lbl=0;
     GetProblemParams(filename, Nrows, Nfeat, Nlabels, lbl_idx, Nlines);
     vector <string> labels;
     
     datos.open(filename, ifstream::in);

     int cmp1, cmp2, cmp = -1;
     bool salirEOF=false;
     while ((!salirEOF)&&(cmp<0))
     {
       if (!getline(datos,cadena)) salirEOF=true;
       cmp1 = (int) cadena.find("{");
       cmp2 = (int) cadena.find("}");
       cmp = cmp1+cmp2;
       //cout << "<-> "  << "   : "<< cmp << " :   " << cadena.c_str() <<endl;
     }
     
     cmp1=cmp1+1;
     cmp2=cmp2-1;
     tmp = cadena.substr(cmp1,cmp2-cmp1+1);
     
     cmp1 = 1;
     while (cmp1>0) 
     {        
	cmp1 = (int) tmp.find(",");
	label = tmp.substr(0,cmp1);
        //  cout << ">labels: " << label.c_str() << endl;
        labels.push_back(label);
	tmp = tmp.substr(cmp1+1,tmp.length()-cmp1);       
	// guardar los labels
     }
     
     cmp=-1;
     while ((!salirEOF)&&(cmp<0))
     {
       if (!getline(datos,cadena)) salirEOF=true;
       cmp1 = strcmp(cadena.c_str(),"@DATA");
       cmp2 = strcmp(cadena.c_str(),"@data");
       cmp = min(abs(cmp1),abs(cmp2));
     }
          
     /* -------------------------------------------------*/
     
     prob.l   = Nrows;             // number of rows
     elements = (Nfeat+1) * Nrows;   // total number of features (nfeat * prob.l)
     
     prob.y  = Malloc(double,prob.l);
     prob.x  = Malloc(struct svm_node *,prob.l);
     x_space = Malloc(struct svm_node,elements);

     /* -------------------------------------------------*/
     
     map<string,int> mapa;
     // map<string,int>::iterator it;
     // it = mapa.find('b');
     
     for (i=0;i< (int) labels.size();i++)
         mapa.insert( pair<string,int>(labels[i],i+1) );       
     
     cmp=1;
     i = 0;
     jj = 0;
     unsigned ii = 0;
     while ((!salirEOF)&&((unsigned)i<Nlines)&&(ii<Nrows))
     {
        if (!getline(datos,cadena)) salirEOF=true;
        cmp = (int) cadena.find(",");
        // recorrer la linea y extraer parametros
                 
        i++;
        if (cmp>=0){
	  
	    prob.x[ii] = &x_space[jj];
           
            tmp = cadena;     
            cmp1 = 1;
            j = 0;
            while (cmp1>=0) 
            {   
                cmp1 = (int) tmp.find(",");
                label = tmp.substr(0,cmp1);                                
                tmp = tmp.substr(cmp1+1,tmp.length()-cmp1);
		
		chomp(label);
    
                if (j==lbl_idx) {
		    /*
                    for (k=0;k<labels.size();k++){
		      //cout << "<"<< labels[k].c_str()<<">" << endl;
                        if (strcmp(labels[k].c_str(),label.c_str())==0) {  
                          lbl = k+1;
                          break;
                        }
                        // cout << "-> "<< lbl << endl;
                    } */ 
		    lbl = mapa[label];
                    prob.y[ii] = lbl;
		    //cout << "<" << label.c_str()<<">" << endl;
                } else {  
                    value = strtod (label.c_str(), NULL);
                    x_space[jj].index = 1+ (int)j;
                    x_space[jj].value = value;
                }
                if (cmp1>=0) {
                   j++;
                   jj++;
		}
            }
            x_space[jj++].index = -1;
            if (j>(Nfeat+1)){     // Nfeat + class
                fprintf(stderr,"Arff read error %s, wrong feature number.\n",filename);
                exit(1); 
            }  
            ii++;
          
        }
     }   
     
     
     //-----------------------------------------
     /*
     for (i=0;i<Nrows;i++){
        for (j=0;j<Nfeat;j++){
	    cout <<  prob.x[i][j].index << ":"<< prob.x[i][j].value << "  "; 
	} 
	cout << prob.x[i][++j].index << " "<<  prob.y[i] << "\n"; 
     }  
     */
     
     datos.close();
     

     return;
}








void ReadProblemArff(const char *filename, unsigned &Nfeat, struct svm_problem &prob, struct svm_node *x_space, vector <string> &etiquetas)
{
     string cadena, tmp, label;
     unsigned jj, j=0;
     int i=0;
     double value;
     ifstream datos;
     size_t elements;
     unsigned Nrows, Nlabels, lbl_idx, Nlines, lbl=0;
     GetProblemParams(filename, Nrows, Nfeat, Nlabels, lbl_idx, Nlines);
     vector <string> labels;
     
     datos.open(filename, ifstream::in);

     int cmp1, cmp2, cmp = -1;
     bool salirEOF=false;
     while ((!salirEOF)&&(cmp<0))
     {
        if (!getline(datos,cadena)) salirEOF=true;
        cmp1 = (int) cadena.find("{");
        cmp2 = (int) cadena.find("}");
        cmp = cmp1+cmp2;
        //cout << "<-> "  << "   : "<< cmp << " :   " << cadena.c_str() <<endl;
     }
     
     cmp1=cmp1+1;
     cmp2=cmp2-1;
     tmp = cadena.substr(cmp1,cmp2-cmp1+1);
     
     cmp1 = 1;
     while (cmp1>0) 
     {        
        cmp1 = (int) tmp.find(",");
        label = tmp.substr(0,cmp1);
        //  cout << ">labels: " << label.c_str() << endl;
        labels.push_back(label);
        tmp = tmp.substr(cmp1+1,tmp.length()-cmp1);       
        // guardar los labels
     }
     
     cmp=-1;
     while ((!salirEOF)&&(cmp<0))
     {
       if (!getline(datos,cadena)) salirEOF=true;
       cmp1 = strcmp(cadena.c_str(),"@DATA");
       cmp2 = strcmp(cadena.c_str(),"@data");
       cmp = min(abs(cmp1),abs(cmp2));
     }
          
     /* -------------------------------------------------*/
     
     prob.l   = Nrows;             // number of rows
     elements = (Nfeat+1) * Nrows;   // total number of features (nfeat * prob.l)
     
     prob.y  = Malloc(double,prob.l);
     prob.x  = Malloc(struct svm_node *,prob.l);
     x_space = Malloc(struct svm_node,elements);

     /* -------------------------------------------------*/
     
     map<string,int> mapa;
     // map<string,int>::iterator it;
     // it = mapa.find('b');
     
     for (i=0;i< (int) labels.size();i++)
         mapa.insert( pair<string,int>(labels[i],i+1) );      
     
     auto search = mapa.find(labels[0].c_str());
     
     cmp=1;
     i = 0;
     jj = 0;
     unsigned ii = 0;
     while ((!salirEOF)&&((unsigned)i<Nlines)&&(ii<Nrows))
     {
        if (!getline(datos,cadena)) salirEOF=true;
        cmp = (int) cadena.find(",");
        // recorrer la linea y extraer parametros
                 
        i++;
        if (cmp>=0){

            prob.x[ii] = &x_space[jj];
           
            tmp = cadena;     
            cmp1 = 1;
            j = 0;
            while (cmp1>=0) 
            {   
                cmp1 = (int) tmp.find(",");
                label = tmp.substr(0,cmp1);                                
                tmp = tmp.substr(cmp1+1,tmp.length()-cmp1);

                chomp(label);
    
                if (j==lbl_idx) {
                    /*
                    for (k=0;k<labels.size();k++){
                        //cout << "<"<< labels[k].c_str()<<">" << endl;
                        if (strcmp(labels[k].c_str(),label.c_str())==0) {  
                          lbl = k+1;
                          break;
                        }
                        // cout << "-> "<< lbl << endl;
                    } */     

                    search = mapa.find(label);                     
                    if(search != mapa.end()) 
                        lbl = search->second;
                    else 
                        lbl = 1;
                
                    // lbl = mapa[label];
                
                    prob.y[ii] = lbl;
                    //cout << "<" << label.c_str()<<">" << endl;
                } else {  
                    value = strtod (label.c_str(), NULL);
                    x_space[jj].index = 1+ (int)j;
                    x_space[jj].value = value;
                }
                if (cmp1>=0) {
                   j++;
                   jj++;
                }
            }
            x_space[jj++].index = -1;
            if (j>(Nfeat+1)){     // Nfeat + class
                fprintf(stderr,"Arff read error %s, wrong feature number.\n",filename);
                exit(1); 
            }  
            ii++;
          
        }
     }   
     
     
     //-----------------------------------------
     /*
     for (i=0;i<Nrows;i++){
        for (j=0;j<Nfeat;j++){
	    cout <<  prob.x[i][j].index << ":"<< prob.x[i][j].value << "  "; 
	} 
	cout << prob.x[i][++j].index << " "<<  prob.y[i] << "\n"; 
     }  
     */
     
     datos.close();
     
     etiquetas = labels;
     

     return;
}

///////////////////////////////////////////////////////////////////////////////////
