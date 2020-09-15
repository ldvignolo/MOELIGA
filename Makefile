#-------------------------------------------------------
# Makefile
#
# running: 
#   export PATH=$PWD:$PATH
#   mpdboot -n 1
#   ./agp npr 2 </dev/null &
# 
# - Dependencias libboost mlpack libensmallen armadillo
# - Ubuntu: sudo apt install libmlpack-dev  libmlpack3 libensmallen-dev libarmadillo9 libarmadillo-dev libboost1.71-dev
# 
#-------------------------------------------------------


all:
	mkdir -p bin
	$(MAKE) -C ./c++/GA all
	mv ./c++/GA/agp bin/
	$(MAKE) -C ./c++/fitness/ all
	mv ./c++/fitness/fitness bin/
	mv ./c++/fitness/test bin/
ubuntu:
	mkdir -p bin
	$(MAKE) -C ./c++/GA all
	mv ./c++/GA/agp bin/
	$(MAKE) -C ./c++/fitness/ ubuntu
	mv ./c++/fitness/fitness bin/
	mv ./c++/fitness/test bin/
ag:
	mkdir -p bin
	$(MAKE) -C ./c++/GA all
	mv ./c++/GA/agp bin/
