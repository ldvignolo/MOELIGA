# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:33:23 2015

@author: mgerard
"""

# LIBRARIES
import re
import numpy
import cPickle as pickle

from sklearn import preprocessing

#from sklearn import grid_search

from sklearn import svm
#from sklearn import ensemble

from sklearn import cross_validation

#from sklearn.cross_validation import KFold
#from sklearn.cross_validation import StratifiedKFold

from sklearn import metrics



#==============================================================================


##===========================================
#def Load_ARFF_File(filename):
#    '''
#    '''
#    
#    # STRUCTURE DATA RETURNED
#    ARFF = {'attributes': list(),'attribute_types': list(), 'class': list(),
#             'data':list(), 'target':list(), 'relation':''}
#    
#    
#    # OPEN FILE
#    F = open(filename, 'rb')
#    FILE = F.read()
#    F.close()
#    
#    # SPLIT DATA INTO HEADER AND DATA
#    header,data = FILE.split('@DATA')
#    
#    
#    # --- HEADER PROCESSING ---
#    ARFF['relation'] = re.findall('@RELATION (.{1,100})\n', header)
#    
#    L = re.findall('@ATTRIBUTE (.{1,100}) (.{1,100})\n', header)
#    
#    for l in L:
#        if l[0] == 'class':
#            ARFF['class'].extend(l[0])
#            
#        else:
#            ARFF['attributes'].append(l[0])
#            ARFF['attribute_types'].append(l[1][:-1])
#    
#    
#    # --- DATA PROCESSING ---
#    if '\r\n' in data:
#        patterns = data.split('\r\n')
#    elif '\n' in data:
#        patterns = data.split('\n')
#    
#    del patterns[0]
#    del patterns[-1]
#    
#    for pattern in patterns:
#        
#        X = list()
#                
#        # SPLIT INTO ATTRIBUTES
#        attributes = pattern.split(',')
#        
#        for attribute in attributes[:-1]:
#            
#            X.append(float(attribute))
#        
#        ARFF['data'].append(X)
#        ARFF['target'].append(attributes[-1])
#    
#    return ARFF
##===========================================


def balanced_cross_validation(labels,Npartitions):
    '''
    '''
    
#    N = 8
#    L = numpy.random.choice(xrange(0,8),90,replace=True)
    N = 0
    C = []
    for c in numpy.unique(labels):
        C.append(numpy.where(labels == c)[0])
        if len(C[-1]) > N:
            N = len(C[-1])
    
    Partitions = []
    
    for k in xrange(0,Npartitions):
        partition = []
        
        for idxs in C:
            
            selected = numpy.random.choice(idxs,N,replace=True)
            
            non_selected = numpy.random.choice(idxs,N,replace=True)
            
            partition.extend([selected,non_selected])
        
        Partitions.append(numpy.array(partition))
    
    return Partitions

#===========================================



#===========================================
def Read_Chromosome(filename):
  '''
  '''
  file = open(filename+'prms.dat','r')
  values = file.readline()
  file.close()
  
  characteristics = [int(value)-1 for value in values.split(',')]
  
  return characteristics


#===========================================  



#===========================================
class ARFF_Object(dict):
    '''
    Maneja los datos del archivo ARFF.
    '''
    
    def __init__(self,arff_filename):
        '''
        '''
        
        # STRUCTURE DATA RETURNED
        self['attributes'] = list()
        self['attribute_types'] = list()
        self['class'] = list()
        self['data'] = list()
        self['target'] = list()
        self['relation'] = ''
        
        self.__name = arff_filename.split('/')[-1].split('.')[0]
        
        
        # OPEN FILE
        F = open(arff_filename, 'rb')
        FILE = F.read()
        F.close()
        
        # SPLIT DATA INTO HEADER AND DATA
        header,data = FILE.split('@DATA')
        
        # --- HEADER PROCESSING ---
        self['relation'] = re.findall('@RELATION (.{1,100})\n', header)
        
        L = re.findall('@ATTRIBUTE (.{1,100}) (.{1,100})\n', header)
        
        for l in L:
            if l[0] != 'class':
                self['attributes'].append(l[0])
                self['attribute_types'].append(l[1][:-1])
        
        
        
        # --- DATA PROCESSING ---
        if '\r\n' in data:
            patterns = data.split('\r\n')
            del patterns[0]
            del patterns[-1]
        elif '\n' in data:
            patterns = data.split('\n')
            del patterns[0]
        
        for pattern in patterns:
            
            X = list()
            
            # SPLIT INTO ATTRIBUTES
            attributes = pattern.split(',')
            
            for attribute in attributes[:-1]:
                
                X.append(float(attribute))
            
            self['data'].append(X)
            self['target'].append(attributes[-1])
        
        self['data'] = numpy.array(self['data'])
        self['target'] = numpy.array(self['target'])
        
        self['class'] = list(set(self['target']))
    
      
    #===========================================
    def size(self):
        return ( len(self['target']) , len(self['data'][0]) )
    
    
    #===========================================
    def get_patterns(self, idxs=[], features=[]):
        '''
        Devuelve el listado de patrones para los patrones indicados en la lista de índices "idx".
        '''
        
        patterns = self['data'][:]
        
        if idxs:
            patterns = patterns[idxs,:][:]
        
        if features:
            patterns = patterns[:,features][:]
        
        return patterns
    
    
    #===========================================
    def get_class(self, idxs=[]):
        '''
        Devuelve el listado de clases para los patrones indicados en la lista de índices "idxs".
        '''
        
        if idxs == []:
            return self['class']
            
        else:
            return [self['class'][idx] for idx in idxs]
    
    
    
    #===========================================
    def class_summary(self,show=False):
        '''
        Devuelve un vector conteniendo el número de patrones por clase.
        '''
        values = [ self['target'].count(x) for x in self['class'] ]
        
        if show == True:
            for ii in xrange(0,len(self['class'])):
                print self['class'][ii] + ': ' + str(values[ii])        
        
        return values
    
        
    #===========================================
    def save(self):
        '''
        Guarda el objeto en un archivo serializado con extensión PKL (pickle).
        '''
        pickle.dump( self, open( self.__name + '.pkl', "wb" ) )
    



#===========================================






#########################
#   CLASIFICADOR
#########################

def classificator(features_file, train_filename, test_filename, validation_filename):
    '''
    '''
    idx_features = []
    
    #=====================
    # LOAD FEATURES
    #=====================
    if features_file != '':
        idx_features = Read_Chromosome(features_file)
    
    
    
    
    #=====================
    # LOAD TRAINING DATA
    #=====================
    name,ext = train_filename.split('/')[-1].split('.')
    if ext == 'arff':
        Train_data = ARFF_Object(train_filename) # ---> LEO DESDE EL ARFF
        
    elif ext == 'pkl':
        Train_data = pickle.load( open( train_filename, "rb" ) )
        
    else:
        print 'Extensión desconocida.'
    
    X_train = Train_data.get_patterns(features=idx_features)
    Y_train = Train_data['target']
    
    
    
    #=====================
    # LOAD TEST DATA
    #=====================
    if test_filename != '':
        
        name,ext = test_filename.split('/')[-1].split('.')
        if ext == 'arff':
            Test_data = ARFF_Object(test_filename) # ---> LEO DESDE EL ARFF
            
        elif ext == 'pkl':
            Test_data = pickle.load( open( test_filename, "rb" ) )
            
        else:
            print 'Extensión desconocida.'
        
        X_test = Test_data.get_patterns(features=idx_features)
        Y_test = Test_data['target']
    
    else:
        X_test = []
        Y_test = []
    
    
    
    #=====================
    # PREPROCESSING --> mean=0.0 - var=1.0
    #=====================
    scaler_train = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler_train.transform(X_train)
    
    if len(X_test) > 0:
        X_test_transformed = scaler_train.transform(X_test)
    else:
        X_test_transformed = []
    
    
    #=====================
    # SET UP CLASSIFIER
    #=====================
    clf = svm.SVC(kernel='linear', C=1, cache_size=500)
    
#    parameters = {'kernel':('linear',), 'C':[1,]}
#    
#    clf = grid_search.GridSearchCV(svm.SVC(cache_size=500), parameters)
#    clf.fit(X_train_transformed, Y_train)
    
#    clf = svm.SVC(kernel='linear', C=1, cache_size=500).fit(X_train_transformed, Y_train)
    #clf = svm.SVC(kernel='poly', degree=3, gamma=0.000075, C=1, cache_size=500).fit(X_train_transformed, Y_train)
    #clf = svm.SVC(kernel='rbf', degree=3, gamma=0.000075, C=1, cache_size=500).fit(X_train_transformed, Y_train)
    
    #clf = ensemble.RandomForestClassifier(50,criterion='entropy')
    #clf = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    
    
    
    #=====================
    # DATA PARTITIONING ---> FALTA MEJORAR!!
    #=====================
    if len(X_test) == 0:
#        kf = cross_validation.KFold(len(X_train_transformed), n_folds=5)
        skf = cross_validation.StratifiedKFold(Y_train,5)
#        Particionado con remestreo ---> Implementar?
        #-------------------------------
        # PARTICIONADO CON BALANCEO
        #-------------------------------
#        bkf = balanced_cross_validation(Y_train,5)
        
        
        #=====================
        # CROSS VALIDATION
        #=====================
    #    CONFUSSION_MATRIX = []
        SCORE = []
        UAR = []
        
        for train_index, test_index in skf:
            
            #======================
            # TRAIN SVM
            #======================
            clf.fit(X_train_transformed[train_index,:],Y_train[train_index])
            
            
            #======================
            # SVM PREDICTION
            #======================
            Y_pred = clf.predict(X_train_transformed[test_index,:])
            
            
            #======================
            # SCORE CALCULATION
            #======================
            score = clf.score(X_train_transformed[test_index,:], Y_train[test_index])
            
            
            #===============================
            # CONFUSSION MATRIX CALCULATION
            #===============================
            M = numpy.float64(metrics.confusion_matrix(Y_train[test_index],Y_pred),labels=Train_data['class'])
            
            
            #===============================
            # UAR CALCULATION
            #===============================
            uar = numpy.diag(M)/numpy.sum(M,1)
            
            
            #===============================
            # SAVE MEASURES
            #===============================
    #        CONFUSSION_MATRIX.append(M)
            SCORE.append(score)
            UAR.append(uar)
        
        
        #========================
        # REPORT
        #========================
        print '=============='
        print '   TRAINING   '
        print '==============\n'
        
        print('SCORE: %.6f +/- %.6f') % (numpy.mean(SCORE),numpy.std(SCORE))
        
        print('UAR: %.6f +/- %.6f\n') % (numpy.mean(UAR),numpy.std(UAR))
    
        #    for confussion_matrix in CONFUSSION_MATRIX:
        #        print '================'
        #        print confussion_matrix
    
    
    else:
        
        #======================
        # TRAIN SVM
        #======================
        clf.fit(X_train_transformed,Y_train)
        
        
        #======================
        # SVM PREDICTION
        #======================
        Y_pred = clf.predict(X_test_transformed)
        
        
        #======================
        # SCORE CALCULATION
        #======================
        score = clf.score(X_test_transformed, Y_test)
        
        
        #===============================
        # CONFUSSION MATRIX CALCULATION
        #===============================
        M = numpy.float64(metrics.confusion_matrix(Y_test,Y_pred),labels=Test_data['class'])
        
        
        #===============================
        # UAR CALCULATION
        #===============================
        uar = numpy.diag(M)/numpy.sum(M,1)
        
        
        #========================
        # REPORT
        #========================
        print '=============='
        print '   TRAINING   '
        print '==============\n'

        print('SCORE: %.6f\n') % (score.flatten())
        
        print('UAR: %.6f +/- %.6f\n') % (numpy.mean(uar),numpy.std(uar))
    
#    for confussion_matrix in CONFUSSION_MATRIX:
#        print '================'
#        print confussion_matrix
    
    
    
    
    if validation_filename != '':
        
        #=====================
        # LOAD TEST DATA
        #=====================
        name,extention = validation_filename.split('/')[-1].split('.')
        if extention == 'arff':
            Validation_data = ARFF_Object(validation_filename) # ---> LEO DESDE EL ARFF
            
        elif extention == 'pkl':
            Validation_data = pickle.load( open( validation_filename, "rb" ) )
            
        else:
            print 'Extensión desconocida.'
        
        
        X_validation = Validation_data.get_patterns(features=idx_features)
        Y_validation = Validation_data['target']
        
        
        
        #=====================
        # PREPROCESSING --> mean=0.0 - var=1.0
        #=====================
        #scaler_test = preprocessing.StandardScaler().fit(X_test)
        X_validation_transformed = scaler_train.transform(X_validation)
        
        
        #========================
        # TRAIN CLASSIFIER
        #========================
        clf.fit(X_train_transformed,Y_train)
        
        
        #======================
        # CLASSIFIER PREDICTION
        #======================
        Y_pred = clf.predict(X_validation_transformed)
        
        
        #======================
        # SCORE CALCULATION
        #======================
        score = clf.score(X_validation_transformed, Y_validation)
        
        
        #===============================
        # CONFUSSION MATRIX CALCULATION
        #===============================
        M = numpy.float64(metrics.confusion_matrix(Y_validation,Y_pred),labels=Validation_data['class'])
        
        
        #===============================
        # UAR CALCULATION
        #===============================
        uar = numpy.diag(M)/numpy.sum(M,1)
        
        
        
        #========================
        # REPORT
        #========================
        print '=============='
        print '   TEST   '
        print '==============\n'
        
        print('SCORE: %.6f') % (score)
        
        print('UAR: %.6f +/- %.6f\n') % (numpy.mean(uar),numpy.std(uar))
        
#        print '============================'
#        print M





  
#==============================================================================
if __name__ == "__main__":
#==============================================================================
    '''
    DESCRIPCION:
    
        -features(Opcional): Indice del archivo que contiene las características
                             a utilizar. Si no se indica, o se indica '', emplea
                             todas las características disponibles.
        
        -train: Especifica el archivo que contiene los patrones que serán usados
                para entrenar el modelo.
        
        -test(Opcional): Especifica el archivo que contiene los patrones que
                         serán usado para evaluar el modelo durante el
                         entrenamiento.
        
        -validation(Opcional): Especifica el archivo que contiene los patrones
                               que serán usado para evaluar el modelo.
        
    
    EJEMPLO:
        
        >> python classificator.py -train iris.arff
        
        >> python classificator.py -features 3 -train GCM_Training.arff -test GCM_Test.arff
        
    '''
    
    
    import sys, timeit
    tic = timeit.default_timer
    

    features_filename = ''
    test_filename = ''
    validation_filename = ''
    
    for ii in xrange(1,len(sys.argv),2):
        
        if sys.argv[ii] == '-features':
            features_filename = sys.argv[ii+1]
        
        elif sys.argv[ii] == '-train':
            train_filename = sys.argv[ii+1]
        
        elif sys.argv[ii] == '-test':
            test_filename = sys.argv[ii+1]
        
        elif sys.argv[ii] == '-validation':
            validation_filename = sys.argv[ii+1]
        
        else:
            print 'Parámetro desconocido.'
    
    
    start = tic()
    classificator(features_filename, train_filename, test_filename, validation_filename)
    stop = tic()
    total = stop - start
    print '--------------------------'
    print('Elapsed time: %.4f seg') % (total)
    print '--------------------------\n\n'