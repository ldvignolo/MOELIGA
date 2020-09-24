import os
import sys
import re

import subprocess

import timeit

import yaml
import itertools


#=====================================================
import argparse

parser = argparse.ArgumentParser(description='Tool to perform multiple experiments with different parameters.')
parser.add_argument('-s', '--experiment_settings', default='runner_settings.yaml', help='Settings file to configure this tool.')
parser.add_argument('-e', '--eliga_settings', default=None, help='Settings file for ELIGA.')
parser.add_argument('-p', '--experiment_path', default='out/', help='Path to output results. By default it is "out/".')
parser.add_argument('-r', '--repetitions', default=10, help='Number of times that the same parameters must be evaluated.')
parser.add_argument('-n', '--notification', action='store_true', help='Telegram notification.')

args = vars(parser.parse_args())
#=====================================================

RUN_ALGORITHM = True  # Esta bandera permite habilitar/deshabilitar la
                      # ejecución del algoritmo. Es útil para hacer pruebas
                      # en la construcción de los paths y manejo de archivos.


#----------------------
# Initializing timer
#----------------------
tic = timeit.default_timer


starting_run_time = timeit.time.strftime("%Y%m%d-%H%M%S")

log_line = 'Starting experiment on {}\n\n'.format(starting_run_time)
print(log_line)
LOG = log_line


#---------------------------------------
# LOAD EXPERIMENT SETTINGS FILE
#---------------------------------
with open(args['experiment_settings'], 'r') as f:
    
    if (sys.version_info.major < 3) or (sys.version_info.minor < 6):
        SETTINGS_RUNNER = yaml.load(f)  # 2.7 and < 3.6
        
    else:
        SETTINGS_RUNNER = yaml.load(f, Loader=yaml.FullLoader)
#---------------------------------------    

names = list(SETTINGS_RUNNER['PARAMETERS'].keys())

L = [SETTINGS_RUNNER['PARAMETERS'][name] for name in names]

experiments = list(itertools.product(*L))


#======================================
# STARTING EXPERIMENT SEQUENCE
#======================================
log_line = 'Starting experiment sequence...\n\n'
print(log_line)
LOG += log_line

EXPERIMENTOS_REALIZADOS = []

for experiment in experiments:
    
    #-------------------------------------------
    if (args['experiment_path'][-1] != '/'):
        folder = args['experiment_path'] + '/'
    else:
        folder = args['experiment_path'] + ''
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    PARAMETERS = dict()
    
    # FILTRO LO QUE NO SE MODIFICA POR FALSE
    USAR = dict()
    
    for k1,v in zip(names, experiment):
        
        if (SETTINGS_RUNNER['DEPENDENCIAS'] is not None) and ((k1 in SETTINGS_RUNNER['DEPENDENCIAS'].keys()) and (v == False)):
            USAR[k1] = True
            
            for k2 in SETTINGS_RUNNER['DEPENDENCIAS'][k1]:
                USAR[k2] = False
                
        else:
            if (k1 not in USAR):
                USAR[k1] = True
        
    #---------------------------------------
    
    
    for k,v in zip(names, experiment):
        
        if USAR[k]:
            
            folder += '{}__{}/'.format(k,v)
            PARAMETERS[k] = v
            
            folder = folder.replace('True', 'true')
            folder = folder.replace('False', 'false')
            
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    
    #########################################################
    # EVITO REPETIR EXPERIMENTOS POR CARPETAS REPETIDAS!!
    #########################################################
    if folder not in EXPERIMENTOS_REALIZADOS:
        
        EXPERIMENTOS_REALIZADOS.append(folder)
        
        
        #-------------------------------------------
        log_line = '\n########################################################\n\n'
        log_line += 'Processing experiment: {}\n'.format(folder)
        print(log_line)
        LOG += log_line
        
        start_initialization = tic()
        
        log_line = 'Starting: {}\n'.format(str(start_initialization))
        log_line += '\n\n============================\n\n\n'
        print(log_line)
        LOG += log_line
        
        
        #-----------------------------------------------------
        # LEO ARCHIVO DE SETTINGS DE ELIGA
        #-----------------------------------------------------
        with open(args['eliga_settings'], 'r') as fp:
            SETTINGS = fp.read()
        
        #-----------------------------------------------------
        # MODIFICO SEGUN LOS PARAMETROS
        #-----------------------------------------------------
        for k,v in PARAMETERS.items():
            SETTINGS = re.sub('\n{}=.*\n'.format(k),
                            '\n{}="{}"\n'.format(k,v) if isinstance(v, str) else '\n{}={}\n'.format(k,v),
                            SETTINGS)
        
        output_folder = folder.replace('True',
                                    'true')
        
        output_folder = output_folder.replace('False',
                                            'false')
        
        SETTINGS = re.sub('\noutdir=.*\n',
                        '\noutdir="{}"\n'.format(output_folder),
                        SETTINGS)
        
        
        #-----------------------------------------------------
        # GUARDO ARCHIVO DE SETTINGS DE ELIGA
        #-----------------------------------------------------
        SETTINGS = ''.join(SETTINGS)
        
        filename_settings = os.path.split(args['eliga_settings'])[-1]
        
        with open(os.path.join(output_folder, filename_settings), 'w') as fp:
            fp.write(SETTINGS)
        
        #-----------------------------------------------------    
        
        
        #------------------------------------
        # CORRER REPLICAS
        #-------------------
        for i in range(1,int(args['repetitions'])+1):
            
            #----------------------
            # CORRER EXPERIMENTO
            #----------------------
            
            # START LOGGING
            start_time = tic()
            log_line = 'Repetition {}/{}\n'.format(str(i), str(args['repetitions']))
            log_line += 'Starting: {}\n'.format(str(start_time))
            print(log_line)
            LOG += log_line
            
            #·····································
            # ALGORITHM
            #·············
            _,eliga_settings = os.path.split(args['eliga_settings'])
            current_path = os.path.join(output_folder, eliga_settings)
            
            if RUN_ALGORITHM:
                os.system('./bin/agp cfg {}'.format(current_path))
            
            
            #·····································
            
            # STOP LOGGING
            stop_time = tic()
            log_line = 'Ending: {}\n'.format(str(stop_time))
            #log_line += '\n----------------------------\n\n'
            print(log_line)
            LOG += log_line
            
            
            log_line = 'Elapsed time: {} sec\n'.format(str(stop_time-start_time))
            log_line += '\n----------------------------\n\n'
            print(log_line)
            LOG += log_line
            
            
            # SAVING SETTINGS FILE
            filename = os.path.split(args['eliga_settings'])[-1]
            with open(folder + filename, 'w') as fp:
                fp.write(SETTINGS)
            
            
        #------------------------------------
        
        
        end_time = tic()
        
        log_line = 'Ending: {}\n\n'.format(str(end_time))
        print(log_line)
        LOG += log_line
    

ending_run_time = timeit.time.strftime("%Y%m%d-%H%M%S")
log_line = 'Finishing on {}\n\n'.format(ending_run_time)
print(log_line)
LOG += log_line



if (args['experiment_path'][-1] != '/'):
    root_folder = args['experiment_path'] + '/'
else:
    root_folder = args['experiment_path'] + ''


#--------------------
# SAVING LOG REPORT
#--------------------
with open('{}/log_{}.txt'.format(root_folder,starting_run_time), 'w') as fp:
    fp.write(LOG)

#--------------------------
# GUARDO UNA COPIA DE C++
#--------------------------
os.system('7z a -t7z -mx=9 {}/c++.7z c++/configs/ c++/fitness/ c++/GA/'.format(root_folder))

#--------------------------------------------
# GUARDO UNA COPIA DE runner_settings.yaml
#--------------------------------------------
os.system('cp {} {}'.format(args['experiment_settings'], root_folder))


#--------------------------------------------
# Notifico
#--------------------------------------------

if (args['notification']):
    import notification
    notification.notify("La instancia de runner " + args['experiment_path'] + " ha finalizado.")
