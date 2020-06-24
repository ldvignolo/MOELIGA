/*
----------------------------------------------------------
EXTREME LEARNING MACHINES FOR C++ version 0.5

Copyright © 2018 Core Invention, Inc. All rights reserved.
Created by SCOTT WILBER, LUIS ARAUJO
----------------------------------------------------------

MATRIX FORM OF ELM
------------------

bias (Lx1)		H (NxL)
_______
       |				f|
       |				 |
       |x1				f|
       |				 |			Σt1 |
X (Nxd)|x2				f|				| T (Nxc)
       |				 |			Σt2 |
       |x3				f|
       |				 |
       |				f|
___________		___________
W (dxL)			B (Lxc)

Matrix dimension variables:
L = # of hidden neurons
N = # of samples
d = # of features (dimensions)
c = # of classes (output)

Matrices/Vectors:
X = Input samples matrix
bias = Bias vector
W = Input weights matrix
H = Hidden layer output matrix
B = Output weights matrix
T = Target matrix

Nodes:
x = input nodes (features/dimensions)
f = hidden layer nodes
t = output nodes


ELM SOLUTION WITH PSEUDO-INVERSE
--------------------------------
Hβ = T, where H (hidden neuron function), β (output weight) and T (target) are matrices.

H = φ(XW + b), where φ is activation function, X is input data Matrix, W is input weight matrix, and b is bias vector.

β = H†T, where H† = inv(H'H)H'		† = pseudo-inverse, ' = transpose

H'H is prone to numerical instabilities if output is close to singular. Therefore,
H† = inv(H'H + αI)H', where α = 50ε, ε = smallest effective increment for double precision. α = 1.1102E-14; I = Identity matrix.

Matrix identity for computation and memory requirements. Faster method for solving  β = H†T

*** Do not Compute β in one line. must break down operations by A and B. ***
Hβ = T
β = H†T
β = H'T
(H'H)β = H'T
(H'H + αI)β = H'T // add regularization constant
// Solve Ax = b using linear least squares systems
A = (H'H + αI)
b = H'T
x = β

*/

#include <cmath>
#include <chrono>
// #include "Eigen-3.3.5/Eigen/Core"
// #include "Eigen-3.3.5/Eigen/QR" 
// #include "Eigen-3.3.5/Eigen/LU"

#include "Eigen-3.3.7/Eigen/Core"
#include "Eigen-3.3.7/Eigen/QR" 
#include "Eigen-3.3.7/Eigen/LU"

using namespace std;
using namespace Eigen;

#ifndef ELM_H
#define ELM_H

// Activation functions
double Sigmoid(double x);
double ReLU(double x);
double Tanh(double x);

// helper functions
int compare(const void *a, const void *b);
MatrixXd buildTargetMatrix(double *Y, int nLabels);
unsigned long long GetTickCount();

#define maxx(a,b) (((a)>(b))?(a):(b))

// entry function to train the ELM model
// INPUT: X, Y, nhn,
// OUTPUT: inW, bias, outW
template <typename Derived>
int elmTrain( double *X, int nfeat, int nsmp, 
		  double *Y,
	      const int nhn, const double C,
	      MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW ) {

	// map the samples into the matrix object: X(Nxd)
	MatrixXd mX = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(X, nsmp, nfeat);
	// normalize matrix/vector
	mX *= 2;
	mX.array() -= 1;
	
	// build target matrix: T(Nxc)
	MatrixXd mTargets = buildTargetMatrix(Y, nsmp); 

	// generate random input weight matrix - inW: W(dxL)
	srand((unsigned int)GetTickCount()); // initialize random seed
	inW = MatrixXd::Random(nfeat, nhn);

	// generate random bias vector B(Lx1)
	bias = MatrixXd::Random(1, nhn); 

	// compute the pre-H matrix
	MatrixXd preH = (mX * inW) + bias.replicate(nsmp, 1);

	// apply activation function to compute hidden neuron output matrix H
	MatrixXd H = preH.unaryExpr(&Sigmoid);

	// compute output weights OutputWeight (beta_i)
	// build matrices to solve Ax = b: (H'H + αI)β = H'T, H† with regularization term
	MatrixXd A = (H.transpose() * H).array() + (MatrixXd::Identity(nhn, nhn)).array()*(1 / C); 
	MatrixXd b = H.transpose() * mTargets;

	// solve the output weights as a solution to a system of linear equations
	outW = A.llt().solve(b);
	
	return 0;
}

// entry function to classify class labels using the trained ELM model on test data
// INPUT : X, inW, bias, outW
// OUTPUT : scores
template <typename Derived>
int elmTest(double *X, int nfeat, int nsmp,
		MatrixBase<Derived> &mScores,
		MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW ) {

	// map the samples into the matrix object: X(Nxd)
	MatrixXd mX = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(X, nsmp, nfeat);
	// normalize matrix/vector
	mX *= 2;
	mX.array() -= 1;

	// build the pre-H matrix
	MatrixXd preH = (mX * inW) + bias.replicate(nsmp, 1);

	// apply the activation function
	MatrixXd H = preH.unaryExpr(&Sigmoid);

	// compute output scores
	mScores = H * outW;

	return 0;
}

// --------------------------
// Activation functions
// --------------------------

double Sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double ReLU(double x)
{
	return maxx(0,x);
}

double Tanh(double x)
{
	return tanh(x);
}

// --------------------------
// Helper functions
// --------------------------

// compares two integer values
int compare(const void *a, const void *b)
{
	const double *da = (const double *)a;
	const double *db = (const double *)b;
	return (*da > *db) - (*da < *db);
}

// builds 1-of-C target matrix from labels array
MatrixXd buildTargetMatrix(double *Y, int nLabels) {

	// make a temporary copy of the labels array
	double *tmpY = new double[nLabels];
	for (int i = 0; i < nLabels; i++) {
		tmpY[i] = Y[i];
	}

	// sort the array of labels
	qsort(tmpY, nLabels, sizeof(double), compare);

	// count unique labels
	int nunique = 1;
	for (int i = 0; i < nLabels - 1; i++) {
		if (tmpY[i] != tmpY[i + 1])
			nunique++;
	}

	delete[] tmpY;

	MatrixXd targets(nLabels, nunique);
	targets.fill(0);

	// fill in the ones
	for (int i = 0; i < nLabels; i++) {
		int idx = (int)(Y[i] - 1); 
		targets(i, idx) = 1;
	}

	// normalize the targets matrix values (-1/1)
	targets *= 2;
	targets.array() -= 1;

	return targets;
}

// simulation of Windows GetTickCount()
unsigned long long GetTickCount()
{
	using namespace std::chrono;
	return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

#endif
