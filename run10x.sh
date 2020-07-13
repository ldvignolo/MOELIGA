#!/bin/bash



for i in {0..10..1}
  do 
     ./agp cfg ./config/leuk/leuk_SETTINGS_$i.cfg
 done


python3 GetAvgResults.py resultados_para_borrar_leuk_PROBANDOSCRIPT/

