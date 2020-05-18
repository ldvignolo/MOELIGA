#-------------------------------------------------------
# Makefile
#
# running: 
#   export PATH=$PWD:$PATH
#   mpdboot -n 1
#   ./agp npr 2 </dev/null &
# 
#   scl enable devtoolset-3 bash
# 
#-------------------------------------------------------


all:
	$(MAKE) -C ./c++/GA all
	mv ./c++/GA/agp ./
	$(MAKE) -C ./c++/fitness/fitness_libsvm/ all
	mv ./c++/fitness/fitness_libsvm/fitsvm ./
	mv ./c++/fitness/fitness_libsvm/testsvm ./
	$(MAKE) -C ./c++/fitness/fitness_python/ all
	mv ./c++/fitness/fitness_python/fitness ./

