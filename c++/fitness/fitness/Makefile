MPICXX=mpicxx
OPCIONES=-g -Wall -std=c++11 -O3 -DMPICH_IGNORE_CXX_SEEK -lpthread -ftree-parallelize-loops=8

all:
	${MPICXX} ${OPCIONES} fitness.cpp ../libsvm-3.20/svm.o -o fitness -lmpich -lc
	g++ ${OPCIONES} test.cpp ../libsvm-3.20/svm.o -o test
