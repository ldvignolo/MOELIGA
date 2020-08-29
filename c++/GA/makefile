#-------------------------------------------------------
# Makefile
#
# running: 
#   export PATH=$PWD:$PATH
#   mpdboot -n 1
#   ./agp npr 2 </dev/null &
#-------------------------------------------------------


CXX=mpicxx
OPCIONES=-g -Wall -std=c++11 -O3  -DMPICH_IGNORE_CXX_SEEK -lpthread -ftree-parallelize-loops=8


all:
	${CXX} ${OPCIONES} agp.cpp -o agp -lmpich -lc
clean:
	rm agp
