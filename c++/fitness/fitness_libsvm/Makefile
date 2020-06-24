MPICXX=mpicxx
OPCIONES=-g -Wall -std=c++11 -O3 -DMPICH_IGNORE_CXX_SEEK

all:
	${MPICXX} ${OPCIONES} fitsvm.cpp ../libsvm-3.20/svm.o -o fitsvm -lmpich -lc
	${MPICXX} ${OPCIONES} testsvm.cpp ../libsvm-3.20/svm.o -o testsvm


