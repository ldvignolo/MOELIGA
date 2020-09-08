# README #

## Version Multi-Objetivo de ELIGA - MOELIGA ##

- TODO
-- Pasar el parametro k del metodo de torneo desde el archivo de configuracion (esta hardcodeado)


Para compilar
- Recordar que MOELIGA debe compilarse con el mismo compilador que MPICH (y mpd)
- Dependencias libboost mlpack libensmallen armadillo
- Ubuntu: sudo apt install libmlpack-dev  libmlpack3 libensmallen-dev libarmadillo9 libarmadillo-dev libboost1.71-dev


Ejecución
- mpdboot
- ./bin/agp npr 6 [cfg archivo_SETTINGS.cfg]
- o para que persista al cerrar la sesion:  ./bin/agp npr 6 </dev/null &
