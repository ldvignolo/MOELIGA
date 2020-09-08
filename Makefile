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
	mkdir -p bin
	$(MAKE) -C ./c++/GA all
	mv ./c++/GA/agp bin/
	$(MAKE) -C ./c++/fitness/ all
	mv ./c++/fitness/fitness bin/
	mv ./c++/fitness/test bin/

