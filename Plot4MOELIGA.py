#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 03 10:08:00 2015

@author: mgerard / modified by lvignolo
"""

import matplotlib.pyplot as plt
import json
from PFrontMOELIGA import getPFronts
import os
import numpy as np

def Plot4MOELIGA(filename):
    
    with open(filename,'r') as fp:
        DATA1 = json.load(fp)
    
    filename, file_extension = os.path.splitext(filename) 
    
    
    # VECTOR OF GENERATIONS
    Generations = range(DATA1['MEASURES']['Generations'])
    
    #===========================
    # LAYOUT
    #===========================
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,9))
    fig.tight_layout(pad=4.0)
    
    PF_DATA = getPFronts(filename+'.txt')
    
    #---------------------------
    # PLOT [1] ---> NFEATURES
    #---------------------------
    absNFEATURES = dict()
    absNFEATURES['elt'] = DATA1['MEASURES']['ELITE']['ABS_Nfeatures'][:-1]
    absNFEATURES['global'] = DATA1['MEASURES']['GENERAL']['ABS_Nfeatures']['mean'][:-1]

    NFEATURES = dict()
    NFEATURES['elt'] = DATA1['MEASURES']['ELITE']['Nfeatures'][:-1]
    NFEATURES['global'] = DATA1['MEASURES']['GENERAL']['Nfeatures']['mean'][:-1]

    UAR = dict()
    UAR['elt'] = DATA1['MEASURES']['ELITE']['UAR'][:-1]
    UAR['global'] = DATA1['MEASURES']['GENERAL']['UAR']['mean'][:-1]

    DIST = dict()
    DIST['elt'] = DATA1['MEASURES']['ELITE']['mDIST'][:-1]
    DIST['global'] = DATA1['MEASURES']['GENERAL']['mDIST']['mean'][:-1]

    axes[0][0].plot(Generations, DIST['elt'],'--r', linewidth=2.0, label='Elite')
    axes[0][0].plot(Generations, DIST['global'],'-b', linewidth=2.0, label='global')
    axes[0][0].set_xlabel(u'Generations', fontsize=14)
    axes[0][0].set_ylabel(u'Mean Distance', fontsize=14)
    axes[0][0].grid()
    axes[0][0].legend()
    axes[0][0].tick_params(axis='x', which='major', labelsize=12)
    axes[0][0].tick_params(axis='y', which='major', labelsize=12)

    axes[0][1].plot(Generations, UAR['elt'],'--r', linewidth=2.0, label='Elite')
    axes[0][1].plot(Generations, UAR['global'],'-b', linewidth=2.0, label='global') 
    axes[0][1].set_xlabel(u'Generations', fontsize=14)
    axes[0][1].set_ylabel(u'UAR', fontsize=14)
    axes[0][1].grid()
    axes[0][1].legend()
    axes[0][1].tick_params(axis='x', which='major', labelsize=12)
    axes[0][1].tick_params(axis='y', which='major', labelsize=12)

    # axes[1][0].plot(Generations, NFEATURES['elt'],'--r', linewidth=2.0, label='Elite') 
    # axes[1][0].plot(Generations, NFEATURES['global'],'-b', linewidth=2.0, label='global') 
    # axes[1][0].set_xlabel(u'Generations', fontsize=14)
    # axes[1][0].set_ylabel(u'Number of features (FUN)', fontsize=14)
    # axes[1][0].grid()
    # axes[1][0].legend()
    # axes[1][0].tick_params(axis='x', which='major', labelsize=12)
    # axes[1][0].tick_params(axis='y', which='major', labelsize=12)

    axes[1][1].plot(Generations, absNFEATURES['elt'],'--r', linewidth=2.0, label='Elite')
    axes[1][1].plot(Generations, absNFEATURES['global'],'-b', linewidth=2.0, label='global') 
    axes[1][1].set_xlabel(u'Generations', fontsize=14)
    axes[1][1].set_ylabel(u'Number of features (ABS)', fontsize=14)
    axes[1][1].grid()
    axes[1][1].legend()
    axes[1][1].tick_params(axis='x', which='major', labelsize=12)
    axes[1][1].tick_params(axis='y', which='major', labelsize=12)
    
    Ngen = len(PF_DATA)-1
    index = np.linspace(0,Ngen,15,dtype=int)
    
    for igen in index:
        OBJ0 = [row[0] for row in PF_DATA[igen]]
        OBJ1 = [row[1] for row in PF_DATA[igen]]
        
        OBJ0, OBJ1 = (list(t) for t in zip(*sorted(zip(OBJ0, OBJ1))))
        OBJ1, OBJ0 = (list(t) for t in zip(*sorted(zip(OBJ1, OBJ0))))
        
        axes[1][0].scatter(OBJ0, OBJ1, label='Gen '+str(igen)) 
        axes[1][0].plot(OBJ0, OBJ1)
    
    axes[1][0].set_xlabel(u'Objetivo 1', fontsize=14)
    axes[1][0].set_ylabel(u'Objetivo 2', fontsize=14)
    axes[1][0].grid()
    axes[1][0].legend()
    axes[1][0].tick_params(axis='x', which='major', labelsize=12)
    axes[1][0].tick_params(axis='y', which='major', labelsize=12)
    
    ###################################
    plt.savefig(filename+'.pdf')
    
    print('Grafico generado exitosamente!\n')
    ###################################




#==============================================================================
if __name__ == "__main__":
#==============================================================================

# python Plot4ELIGA.py xxx.json yyy.json

    import sys

    filename1 = sys.argv[1]
    print(filename1)
    
    if len(sys.argv)>2:
        filename2 = sys.argv[2]
        print(filename2)
        Plot4MOELIGA(filename1, filename2)
    else:    
        Plot4MOELIGA(filename1)


