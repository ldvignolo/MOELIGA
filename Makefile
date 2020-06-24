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
	$(MAKE) -C ./c++/fitness/fitness/ all
	mv ./c++/fitness/fitness/fitness ./
	mv ./c++/fitness/fitness/test ./
svm:
	$(MAKE) -C ./c++/fitness/fitness_libsvm/ all
	mv ./c++/fitness/fitness_libsvm/fitsvm ./
	mv ./c++/fitness/fitness_libsvm/testsvm ./
elm:
	$(MAKE) -C ./c++/fitness/fitness_elm/ all
	mv ./c++/fitness/fitness_elm/fitness ./
	mv ./c++/fitness/fitness_elm/testelm ./
