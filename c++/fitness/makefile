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



# OPCIONESF=-g -Wall -O3 -I/usr/include -L/usr/local/lib64/ -fopenmp -larmadillo -lboost_serialization -lmlpack -Wl,-rpath=/usr/local/lib64/ -llapack -lblas -larpack 
# OPCIONESU=-g -Wall -O3 -I/usr/include -L/usr/local/lib64/ -fopenmp -larmadillo -lboost_serialization -lmlpack -Wl,-rpath=/usr/lib/x86_64-linux-gnu/ -llapack -lblas -larpack 

OPCIONESF=-g -Wall -O3 -fopenmp -larmadillo -lboost_serialization -lmlpack -Wl,-rpath=/usr/local/lib64/ -llapack -lblas -larpack 
OPCIONESU=-g -Wall -O3 -fopenmp -larmadillo -lboost_serialization -lmlpack -Wl,-rpath=/usr/lib/x86_64-linux-gnu/ -llapack -lblas -larpack 

CXX=g++
CXY=mpicxx
OPCIONESM= -std=c++11 -DMPICH_IGNORE_CXX_SEEK -lpthread -ftree-parallelize-loops=8 -lmpich 


all:
	${CXX} test.cpp -o test ${OPCIONESF} -lc 
	${CXY} fitness.cpp -o fitness ${OPCIONESM} ${OPCIONESF} 
test:
	${CXX} test.cpp -o test ${OPCIONESF} -lc
fit:
	${CXY} fitness.cpp -o fitness ${OPCIONESF} ${OPCIONESM} 
ubuntu:
	${CXX} test.cpp -o test ${OPCIONESU} -lc
	${CXY} fitness.cpp -o fitness ${OPCIONESM} ${OPCIONESU}


# mpicxx fitness.cpp -o fitness -DMPICH_IGNORE_CXX_SEEK -lpthread -ftree-parallelize-loops=8 -lmpich -g -Wall -O3 -fopenmp -larmadillo -lboost_serialization -lmlpack -Wl,-rpath=/usr/lib/x86_64-linux-gnu/ -llapack -lblas -larpack -lc
