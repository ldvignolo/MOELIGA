# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:26:02 2015

@author: mgerard
"""

##===========================================


def ARFFbuilder(data_file, label_file, name):
    
    #----------------------------
    # IMPORTING MODULES
    import numpy
    
    
    #----------------------------
    # LOAD LABELS
    #----------------------------
    print "\nLoading LABELS..."
    with open(label_file, 'r') as fp:
        LABELS = fp.readlines()
    #----------------------------
    
    
    
    #----------------------------
    # LOAD DATA
    #----------------------------
    print "\nLoading DATA..."
    with open(data_file, 'r') as fp:
        DATA = fp.readlines()
    
    
    #==================================
    # MAKE HEADING
    #==================================
    FILE = '@RELATION ' + name + '\n'
    
    
    #==================================
    # DEFINITION OF ATTRIBUTES
    #==================================
    DATASET = ''
        
    N = 0
    for idx in xrange(len(DATA)):
        dataset = ''
        data = DATA[idx].split(' ')[:-2]
        if N == 0:
            N = len(data)
        
        dataset += ','.join(str(d)+'.0' for d in data) + ',' + LABELS[idx]
        DATASET += dataset
    
    for idx in xrange(N):
        FILE += '@ATTRIBUTE F' + str(idx) + ' NUMERIC\n'
        
        
        
        
    
    
    #==================================
    # DEFINITION OF CLASSES
    #==================================
    labels = list(set(LABELS))
    L = ",".join(str(int(label)) for label in labels)
    FILE += '@ATTRIBUTE class {' + L + '}\n\n'
    
    
    #==================================
    # DEFINITION OF CLASSES
    #==================================
    FILE += '@DATA\n'
    FILE += DATASET
    
    
    
    #==================================
    # SAVING ARFF FILE
    #==================================
    print "\nSaving ARFF file..."
    with open(name + '.arff', 'w') as fp:
        fp.writelines(FILE)
    
    
    
    


#==============================================================================
if __name__ == "__main__":
#==============================================================================
    
    # EXAMPLE:
    #
    # > python ARFFbuilder.py -data gisette_train.data -label gisette_train.labels -name TRAIN
    
    
    import sys
    
    data = 'data'
    label = 'labels'
    name = 'example'
    help = False
    
    for ii in xrange(1,len(sys.argv),2):
        
        if sys.argv[ii] == '-data':
            data = sys.argv[ii+1]
        
        elif sys.argv[ii] == '-label':
            label = sys.argv[ii+1]
        
        elif sys.argv[ii] == '-name':
            name = sys.argv[ii+1]
        
        elif sys.argv[ii] == '-help':
            print('\nExample:\n\n > python ARFFbuilder.py -data gisette_train.data -label gisette_train.labels -name gisette_train\n')
            help = True
            
        else:
            print 'Parámetro desconocido.'
    
    if not help:
        ARFFbuilder(data, label, name)
