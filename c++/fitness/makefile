#-------------------------------------------------------
# Makefile
#
# Pre-requisitos:
# sudo dnf install boost-devel armadillo-devel
# 
# - Compilar mlpack
#   $ mkdir build
#   $ cd build
#   $ cmake ../
#   $ sudo make install
# 
#   export LD_LIBRARY_PATH="/usr/local/lib64/:$LD_LIBRARY_PATH"
# 
#-------------------------------------------------------

# -std=c++11 -I../src/  -L/usr/lib64 -I/usr/include/armadillo_bits/
# 

CXX=g++
CXY=mpicxx
OPCIONES=-g -Wall -O3 -I/usr/include -L/usr/local/lib64/ -fopenmp -larmadillo -lboost_serialization -lmlpack -Wl,-rpath=/usr/local/lib64/
OPCIONES2= -std=c++11 -O3  -DMPICH_IGNORE_CXX_SEEK -lpthread -ftree-parallelize-loops=8


all:
	${CXX} ${OPCIONES} test.cpp -o test -lc
	${CXY} ${OPCIONES} ${OPCIONES2} fitness.cpp -o fitness -lmpich -lc
test:
	${CXX} ${OPCIONES} test.cpp -o test -lc
fit:
	${CXY} ${OPCIONES} ${OPCIONES2} fitness.cpp -o fitness -lmpich -lc
