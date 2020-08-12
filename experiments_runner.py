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
parser.add_argument('-s', '--experiment_settings', default='SETTINGS.yaml', help='Settings file to configure this tool.')
parser.add_argument('-e', '--eliga_settings', default=None, help='Settings file for ELIGA.')
parser.add_argument('-p', '--current_path', default='out/', help='Path to output results. By default it is "out/".')
parser.add_argument('-r', '--repetitions', default=10, help='Number of times that the same parameters must be evaluated.')

args = vars(parser.parse_args())
#=====================================================



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
        SETTINGS = yaml.load(f)  # 2.7 and < 3.6
        
    else:
        SETTINGS = yaml.load(f, Loader=yaml.FullLoader)
#---------------------------------------    

names = list(SETTINGS.keys())

L = [SETTINGS[name] for name in names]

experiments = list(itertools.product(*L))


#======================================
# STARTING EXPERIMENT SEQUENCE
#======================================
log_line = 'Starting experiment sequence...\n\n'
print(log_line)
LOG += log_line

for experiment in experiments:
    
    #-------------------------------------------
    folder = args['current_path'] + ''
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    PARAMETERS = dict()
    for k,v in zip(names, experiment):
        folder += '{}__{}/'.format(k,v)
        PARAMETERS[k] = v
        
        folder = folder.replace('True', 'true')
        folder = folder.replace('False', 'false')
        
        if not os.path.exists(folder):
            os.makedirs(folder)
    
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
        SETTINGS = re.sub('\n{}=.*\n'.format(k), '\n{}="{}"\n'.format(k,v) if isinstance(v, str) else '\n{}={}\n'.format(k,v), SETTINGS)
    
    output_folder = folder.replace('True', 'true')
    output_folder = output_folder.replace('False', 'false')
    SETTINGS = re.sub('\noutdir=.*\n', '\noutdir="{}"\n'.format(output_folder), SETTINGS)
    
    
    #-----------------------------------------------------
    # GUARDO ARCHIVO DE SETTINGS DE ELIGA
    #-----------------------------------------------------
    SETTINGS = ''.join(SETTINGS)
    with open(args['eliga_settings'], 'w') as fp:
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
        
        os.system('./agp cfg {}'.format(args['eliga_settings']))
        
        
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
    
    os.system('python3 GetAvgResults.py '+ output_folder)

ending_run_time = timeit.time.strftime("%Y%m%d-%H%M%S")
log_line = 'Finishing on {}\n\n'.format(ending_run_time)
print(log_line)
LOG += log_line


#--------------------
# SAVING LOG REPORT
#--------------------
with open('log_{}.txt'.format(starting_run_time), 'w') as fp:
    fp.write(LOG)
