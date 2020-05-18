#-------------------------------------------------------
# Makefile
#
# running: 
#   export PATH=$PWD:$PATH
#   mpdboot -n 1
#   ./agp npr 2 </dev/null &
#-------------------------------------------------------

CXX=mpicxx
OPCIONES=-g -Wall -O3 -I../util -I/usr/include -I/usr/local/mpich2/include -DMPICH_IGNORE_CXX_SEEK
BIBLIOTECAS=-lm -I../olvq -I../../tofysrc
MPI_HOME := /usr/local/mpich2


all:
	${CXX} ${OPCIONES} fitness.cpp -o fitness ${BIBLIOTECAS} -L$(MPI_HOME)/lib -lmpich -lc
clean:
	rm fitness
