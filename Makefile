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

