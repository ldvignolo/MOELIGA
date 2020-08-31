/*
main.cpp
SLFN

Created by LUIS ARAUJO on 12/6/18.
Copyright © 2018 Core Invention, Inc. All rights reserved.

Extreme Learning Machines (ELM) - Single hidden layer feed forward neural (SLFN) network.
reference: Huang, G.-B., What are Extreme Learning Machines? 2015. http://www.ntu.edu.sg/home/egbhuang/pdf/ELM-Rosenblatt-Neumann.pdf
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "Elm.h"

// Read/Write data functions
bool ReadFile(std::string filename, vector<double> &vec);
bool WriteFile(std::string filename, MatrixXd &mData);

double ScoreAccuracy(MatrixXd &resData, vector<double> &targetVec);
double ScoreUAR(MatrixXd &resData, vector<double> &targetVec, bool mshow);


// read data from file
bool ReadFile(std::string filename, vector<double> &vec) {

    fstream fs;

    string filepath = filename;

    fs.open(filepath.c_str(), std::ios_base::in);
    if (!fs) {
        return false;
    }

    double n;

    while (fs >> n) {
        vec.push_back(n);
    }

    return true;
}

// write data to file
bool WriteFile(std::string filename, MatrixXd &mData) {

    std::ofstream os;

    string filepath = filename;

    os.open(filepath.c_str(), std::ios_base::out);
    if (!os) {
        return false;
    }

    os << mData << endl;
    os.close();
    
    return true;
}



//Calculate the training/testing score accuracy
double ScoreAccuracy(MatrixXd &resData, vector<double> &targetVec) {

    double match = 0;

    // determine the col index with max row val 
    int nRows = (int) resData.rows();
    Index maxIndex;
    VectorXd maxVal(resData.rows());
    for (int i = 0; i < nRows; ++i) {
        maxVal(i) = resData.row(i).maxCoeff(&maxIndex);

        if ((double)(maxIndex + 1) == targetVec[i])
            match += 1;
    }
    
    double accuracy = (match / targetVec.size()); // * 100.;

    return accuracy;
}

double ScoreUAR(MatrixXd &resData, vector<double> &targetVec, bool mshow) {

    // determine the col index with max row val 
    int nRows = (int) resData.rows();
    Index maxIndex;
    VectorXd maxVal(resData.rows());
    
    vector <int> tmp;
    bool neu;
    for (int i=0;i<(int)targetVec.size();i++){
        neu = true;     
        for (int k=0;k<(int)tmp.size();k++)
        if (targetVec[i]==tmp[k]) { neu = false; break; }        
        if (neu) tmp.push_back(targetVec[i]);
    }
    short nclass = tmp.size();
    int MC[nclass][nclass] = {{0}};
    memset(MC,0,nclass*nclass*sizeof(int));    
    
    
    
    for (int i = 0; i < nRows; ++i) {
        maxVal(i) = resData.row(i).maxCoeff(&maxIndex);

        MC[ ((int) targetVec[i]-1) ][ maxIndex ]++;
    }
    
    
    double UAR=0, rr;
    for (short i=0;i<nclass;i++){
        rr = 0;
        for (short k=0;k<nclass;k++)
        rr = rr + MC[ i ][ k ];
        UAR = UAR + MC[ i ][ i ]/rr;
    }
    UAR = UAR/nclass;
    
    if (mshow) 
    {
        cout << "                             \"CONFUSION_MATRIX\": " << "[" ;
        
	    for (short i=0;i<nclass;i++)
        {
          if (i==0) cout << "[";    
          else cout << "                                                  " << "[" ;            
            
	      for (short k=0;k<nclass;k++) {
		      cout << MC[i][k]; 
              if (k<(nclass-1)) cout << ", ";  
          }
          cout << "]";    
          if (i<(nclass-1)) cout << "," << endl;  
        }
        cout << "]," << endl;   
    }
    
    
    
    return UAR;
}





          











