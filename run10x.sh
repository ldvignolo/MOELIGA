#!/bin/bash



for i in {0..10..1}
  do 
     ./agp cfg ./config/leuk/leuk_SETTINGS_1.cfg
     ./agp cfg ./config/leuk/leuk_SETTINGS_2.cfg
     ./agp cfg ./config/leuk/leuk_SETTINGS_3.cfg
     ./agp cfg ./config/leuk/leuk_SETTINGS_4.cfg
 done


python3 GetAvgResults.py resultados_para_borrar_leuk_PROBANDOSCRIPT/

