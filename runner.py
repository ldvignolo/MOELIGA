import os
import sys
import re

import subprocess

import timeit

import yaml
import itertools

import json

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
# BUILDING EXPERIMENT SEQUENCE
#======================================
EXPERIMENTOS = []
FOLDERS = []
for experiment in experiments:
    
    #-------------------------------------------
    if (args['experiment_path'][-1] != '/'):
        folder = args['experiment_path'] + '/'
    else:
        folder = args['experiment_path'] + ''
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    parameters = dict()
    
    # FILTRO LO QUE NO SE MODIFICA POR FALSE
    USAR = dict()
    
    for name,value in zip(names,experiment):
        
        # NO CARGUE PREVIAMENTE EL VALOR
        if name not in USAR:
            
            USAR[name] = True
            
            if (name in SETTINGS_RUNNER['DEPENDENCIAS']) and (value == False):
                
                for key in SETTINGS_RUNNER['DEPENDENCIAS'][name]:
                    
                    USAR[key] = False
        
    #---------------------------------------
    
    
    for k,v in zip(names, experiment):
        
        if USAR[k]:
            
            folder += '{}__{}/'.format(k,v)
            parameters[k] = v
            
            folder = folder.replace('True', 'true')
            folder = folder.replace('False', 'false')
            
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    
    #########################################################
    # EVITO REPETIR EXPERIMENTOS POR CARPETAS REPETIDAS!!
    #########################################################
    if folder not in FOLDERS:
        EXPERIMENTOS.append([folder, parameters])
        FOLDERS.append(folder)



if (args['experiment_path'][-1] != '/'):
    folder = args['experiment_path'] + '/'
else:
    folder = args['experiment_path'] + ''

with open(os.path.join(folder,'EXPERIMENTOS.temp'), 'w') as fp:
    json.dump(EXPERIMENTOS, fp, indent=4)

#with open(os.path.join(folder,'FOLDERS.temp'), 'w') as fp:
    #fp.write('\n'.join(FOLDERS))


base_folder = folder

del FOLDERS
del EXPERIMENTOS


    
###########################################
# COMIENZO LA SECUQNCIA DE EXPERIMENTOS
###########################################

# AGREGAR ACA CODIGO PARA QUE LEA "EXPERIMENTOS.temp"
# Y RETOME LA SECUENCIA (pisa replicas)

with open(os.path.join(base_folder,'EXPERIMENTOS.temp'), 'r') as fp:
    EXPERIMENTS = json.load(fp)


# NUMERO TOTAL DE EXPERIMENTOS
N = len(EXPERIMENTS) * int(args['repetitions'])
    

log_line = 'Starting experiment sequence...\n\n'
print(log_line)
LOG += log_line

n = 0
#for (folder,PARAMETERS) in EXPERIMENTOS:
while EXPERIMENTS:
    
    folder,PARAMETERS = EXPERIMENTS.pop(0)
    
    #-------------------------------------------
    log_line = '\n########################################################\n\n'
    log_line += 'Processing experiment: {}\n'.format(folder)
    print(log_line)
    LOG += log_line
    
    start_initialization = tic()
    
    log_line = 'Starting: {}\n'.format(str(start_initialization))    
    log_line += '\n\n============================\n\n\n'
    # print(log_line)
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
        
        n += 1  # EXPERIMENTO/TOTAL EXPERIMENTOS
        
        #----------------------
        # CORRER EXPERIMENTO
        #----------------------
        
        # START LOGGING
        start_time = tic()
        log_line = 'Repetition {}/{} [{}/{}]\n'.format(str(i), str(args['repetitions']), n, N)
        print(log_line)
        log_line += 'Starting: {}\n'.format(str(start_time))
        # print(log_line)
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
        #print(log_line)
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
    #print(log_line)
    LOG += log_line
    
    
    
    # SALVO EXPERIMENTOS
    with open(os.path.join(base_folder,'EXPERIMENTOS.temp'), 'w') as fp:
        json.dump(EXPERIMENTS,fp, indent=4)
    
    with open(os.path.join(base_folder,'EXPERIMENTOS.temp'), 'r') as fp:
        EXPERIMENTS = json.load(fp)
    

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
os.system('7z a -t7z -mx=9 {}c++.7z c++/configs/ c++/fitness/ c++/GA/'.format(root_folder))

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
