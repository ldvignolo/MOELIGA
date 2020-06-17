MPICXX=mpicxx
OPCIONES=-g -Wall -std=c++11 -O3 -I../util -I/usr/include -I/usr/local/mpich2/include -DMPICH_IGNORE_CXX_SEEK

all:
	${MPICXX} ${OPCIONES} fitness.cpp ../libsvm-3.20/svm.o -o fitness -lmpich -lc
	g++ ${OPCIONES} testelm.cpp ../libsvm-3.20/svm.o -o testelm
