# README #

## Version Multi-Objetivo de ELIGA - MOELIGA ##

- TODO
-- Pasar el parametro k del metodo de torneo desde el archivo de configuracion (esta hardcodeado)


Para compilar
- Recordar que MOELIGA debe compilarse con el mismo compilador que MPICH (y mpd)


Ejecución
- `>> mpdboot`
- `>> ./agp cfg archivo_SETTINGS.cfg`
- o para que persista al cerrar la sesion:  - `>> ./agp </dev/null &`




---

Compilar MPICH en un path sin espacios (**No hace falta compilar como root, pero si se hace**)

- `>> ./configure`
- `>> make`
- `>> make install` (ejecutar como *root*)

**NOTA** Si falla `sort`, es necesario instalar coreutils (FC25)


Luego ejecutar:

- `>> cd src/pm/mpd`
- `>> ./configure`
- `>> make`
- `>> make install` (ejecutar como *root*)

Crear archivo:

>> vim /home/mgerard/.mpd.conf (poner secretword=<...>)

>> chmod 600 /home/mgerard/.mpd.conf




Luego, para probar (en cualquier lugar):

`>> mpdboot` (no debería haber salida si funciona correctamente)

Para saber si levanta el demonio:

`>> mpdtrace` (debería dar el nombre del host si funciona correctamente)

`>> mpdallexit` (mata el proceso)
