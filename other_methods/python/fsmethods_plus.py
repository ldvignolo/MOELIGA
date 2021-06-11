# pip install skrebate    
# pip install ReliefF
# pip install arff
# pip install mlxtend

import os
import json
import time
import sys
import pickle
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TuRF
from ReliefF import ReliefF as ReliefF2
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
#from sklearn.feature_selection import SequentialFeatureSelector, f_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sys.path.append('../../')
import notification


#---------------------------------------------------------


def estandarizar(X_train, X_test):
    # Estandarización de los datos (sobre cada fold)
    scaler = StandardScaler()
    # Ajuste sobre de train
    scaler.fit(X_train)
    # transformación de datos train y test
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test
    
    
def estandarizar(X_train):
    # Estandarización de los datos (sobre cada fold)
    scaler = StandardScaler()
    # Ajuste sobre de train
    scaler.fit(X_train)
    # transformación de datos train y test
    X_train = scaler.transform(X_train)
    
    return X_train    


#---------------------------------------------------------


def loadDataset(file1,file2,encode=True):
    trnData, trnLabels = parseArff(file1)
    tstData, tstLabels = parseArff(file2)
    
    if (encode):
        lb_encoder = LabelEncoder()
        lb_encoder.fit(trnLabels)
        trnLabels = lb_encoder.transform(trnLabels)        
        tstLabels = lb_encoder.transform(tstLabels)

    return trnData, tstData, trnLabels, tstLabels


def loadDataset(file1, encode=True):
    trnData, trnLabels, header = parseArff(file1)
    
    if (encode):
        lb_encoder = LabelEncoder()
        lb_encoder.fit(trnLabels)
        trnLabels = lb_encoder.transform(trnLabels)
  
    return trnData, trnLabels, header
    

def parseArff(infile):
    
    data, meta = arff.loadarff(infile)
    nparray = pd.DataFrame(data).to_numpy()

    data = nparray[:,:-1]
    labels = nparray[:,-1] 
    header = meta.names()
    header = header[:-1]
    
    return data, labels, header


def loadJSON(jfile,tiempo,nf,nb):
    
    with open(jfile) as json_file:
        data = json.load(json_file)
        mydict = {}    
        CLASIFICADORES = data[0]['CLASIFICADORES']            
        for idx_r,classifier in enumerate(CLASIFICADORES.keys()):                
            uar = data[0]['CLASIFICADORES'][classifier]['UAR']                   
            mydict.update({classifier:uar})
    mydict.update({'Elapsed Time':tiempo})        
    mydict.update({'No. of Features':nf})
    mydict.update({'No. of Neighbors':nb})
    return mydict



def batchRelief2(file1, nf, methods, encodeLabels=True, nb=20, mpath='None', dataset='None', tag='', confpath='settings/', testbin='bin/test'):
    
    if nf==0:
        return
    
    print('\n < '+dataset+' >')

    prevpath=os.getcwd()
    results = {}
    
    if not os.path.exists(prevpath + '/_resultados_' + tag + '/'):
        os.makedirs(prevpath + '/_resultados_' + tag + '/')
  
    trnData, trnLabels, header = loadDataset(file1, encodeLabels)
    trnData = estandarizar(trnData)

    num_examples = len(trnLabels)
    num_features = trnData.shape[0]

    if num_examples<nb:
        nb = num_examples-1
        
    # # #        

    fsmethod = 'sk_ReliefF'    
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:                
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            fs = ReliefF(n_features_to_select=nf, n_neighbors=nb)
            fs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = fs.top_features_[:nf]
            fv.sort()
            fv = [x+1 for x in fv] # 0 based 
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_' + fsmethod + '.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
                
        tmpdict = loadJSON(jsonfile,etime,nf,nb)
        results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_SURF'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:

        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            fs = SURF(n_features_to_select=nf)
            fs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = fs.top_features_[:nf]
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_' + fsmethod + '.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
                    
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
                
        tmpdict = loadJSON(jsonfile,etime,nf,nb)
        results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_SURFstar'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:        
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            fs = SURFstar(n_features_to_select=nf)
            fs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = fs.top_features_[:nf]
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_' + fsmethod + '.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,nb)
        results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_MultiSURF'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:  
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:    
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            fs = MultiSURF(n_features_to_select=nf)
            fs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = fs.top_features_[:nf]
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_' + fsmethod + '.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,nb)
        results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_MultiSURFstar'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:        
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            fs = MultiSURFstar(n_features_to_select=nf)
            fs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = fs.top_features_[:nf]
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_' + fsmethod + '.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
                    
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,nb)
        results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_TuRF'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:        
            
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            fs = TuRF(core_algorithm="ReliefF", n_features_to_select=nf)
            fs.fit(trnData, trnLabels, fit_params={'headers': header})
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = fs.top_features_[:nf]
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_' + fsmethod + '.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
                    
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,nb)
        results.update({fsmethod:tmpdict})
    
    # # #
   
    fsmethod = 'py_ReliefF'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            if nb<1:
                inb = round(len(trnLabels)*nb)
            else:
                inb = nb        
            fs = ReliefF2(n_neighbors=inb, n_features_to_keep=nf)
            fs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = fs.top_features[:nf]
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_'+ fsmethod +'.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
                    
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,nb)
        results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'MutualInfo'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            mi_features = mutual_info_classif(trnData, trnLabels, random_state=42)
            fv = np.argsort(mi_features)[-1:0:-1]
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = fv[:nf]
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_'+ fsmethod +'.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,'-')
        results.update({fsmethod:tmpdict})
    
    # # # 
    
    fsmethod = 'RFE-DT'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            dtree = DecisionTreeClassifier(random_state=0, max_depth=2)        
            rfe = RFE(dtree, n_features_to_select=nf, step=1)
            rfe.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = rfe.get_support(indices=True)
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_'+ fsmethod +'.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()

        tmpdict = loadJSON(jsonfile,etime,nf,'-')
        results.update({fsmethod:tmpdict})
    
    # # # 
    # Univariate feature selection with F-test for feature scoring
    fsmethod = 'KBest-Ftest'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:        
            
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            kb = SelectKBest(f_classif, k=nf)
            kb.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = kb.get_support(indices=True)
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_'+ fsmethod +'.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,'-')
        results.update({fsmethod:tmpdict})    
    
    
    # # # 
    # SFS - DT
    fsmethod = 'SFS-DT'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:

        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            dtree = DecisionTreeClassifier(random_state=0, max_depth=None)
            sfs = SFS(dtree, k_features=nf, forward=True, floating=False, verbose=0, scoring='balanced_accuracy', cv=3, n_jobs=-1)
            sfs = sfs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            fv = np.array(sfs.k_feature_idx_)
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_'+ fsmethod +'.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
                        
        tmpdict = loadJSON(jsonfile,etime,nf,'-')
        results.update({fsmethod:tmpdict})    
    
    
    # # # 
    # SFFS - DT
    fsmethod = 'SFFS-DT'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            dtree = DecisionTreeClassifier(random_state=0, max_depth=None)
            sfs = SFS(dtree, k_features=nf, forward=True, floating=True, verbose=0, scoring='balanced_accuracy', cv=3, n_jobs=-1)
            sfs = sfs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = np.array(sfs.k_feature_idx_)
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_'+ fsmethod +'.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,'-')
        results.update({fsmethod:tmpdict})    
    
    
    # # # 
    # SBS - DT
    fsmethod = 'SBS-DT'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:        
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            dtree = DecisionTreeClassifier(random_state=0, max_depth=None)
            sfs = SFS(dtree, k_features=nf, forward=False, floating=False, verbose=0, scoring='balanced_accuracy', cv=3, n_jobs=-1)
            sfs = sfs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = np.array(sfs.k_feature_idx_)
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_'+ fsmethod +'.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()
            
            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,'-')
        results.update({fsmethod:tmpdict})     
    
    
    # # # 
    # SBFS - DT
    fsmethod = 'SBFS-DT'
    
    jsonfile = prevpath + '/_resultados_' + tag + '/' + dataset + '_' + fsmethod + '_mlpack.test'
    time_file = prevpath + '/_resultados_' + tag + '/' + dataset+'_ElapsedTime_' + fsmethod + '.txt'
    if fsmethod in methods:
        
        if os.path.isfile(jsonfile) and os.path.isfile(time_file):            
            
            pickle_in = open(time_file, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
            etime = data['elapsed_time']
            
        else:        
        
            print(fsmethod, end=' ', flush=True)
            start_time = time.time()
            dtree = DecisionTreeClassifier(random_state=0, max_depth=None)
            sfs = SFS(dtree, k_features=nf, forward=False, floating=True, verbose=0, scoring='balanced_accuracy', cv=3, n_jobs=-1)
            sfs = sfs.fit(trnData, trnLabels)
            elapsed_time = time.time() - start_time
            etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            
            fv = np.array(sfs.k_feature_idx_)
            fv.sort()
            fv = [x+1 for x in fv] # 0 based
            featsfile = prevpath + '/_resultados_' + tag + '/' + dataset+'_features_'+ fsmethod +'.txt'
            f=open(featsfile,'w')
            for ele in fv:
                f.write(str(ele)+' ')
            f.write('\n')    
            f.close()

            cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
            os.chdir(mpath)
            os.system(cmd)
            os.chdir(prevpath)
            
            pickle_out = open(time_file, "wb")
            pickle.dump({"elapsed_time": etime}, pickle_out,)
            pickle_out.close()
        
        tmpdict = loadJSON(jsonfile,etime,nf,'-')
        results.update({fsmethod:tmpdict})      
    
    print('', flush=True)
    
    return results


def addsheet(dtset,tmpres,sheets,writer,workbook):
    
    df = pd.DataFrame(tmpres)
    df.to_excel(writer, sheet_name='{}'.format(dtset), header=True, startcol=0)
    sheets.append(pd.DataFrame(df, columns=df.columns))
    #sheets[-1:].to_excel(writer, sheet_name='{}'.format(dtset), header=True, startcol=0)
    worksheet = writer.sheets[dtset]   
    col_width=0
    for idx, col in enumerate(df.index):  
        if (len(col)>col_width):
            col_width = len(col)+4        
    worksheet.set_column('A:A', col_width)   
    formato = workbook.add_format({'num_format': '#,####0.0000', 'align': 'center'})
    for idx, col in enumerate(df):  
        col_width = len(col)+4
        j=idx+1
        worksheet.set_column(j, j, col_width,formato)   
    
    return sheets, writer

#---------------------------------------------------------

 #               MOELIGA - NFEATS
 #
 #                        |  2nd opt  | 1rst opt
 #   dermatology          |     5     |     6
 #   optdigits            |           |    10
 #   movement             |     8     |     9
 #   arrhythmia           |    12     |    11
 #   madelon              |    24     |    22
 #   smartphone-activity  |           |     8
 #   isolet               |    37     |    44
 #   mfeat                |    17     |    14
 #   leukemia             |    30     |    30
 #   all-leukemia         |   145     |   168
 #   yeoh                 |   298     |   270
 #   gcm                  |   241     |   264
 #   tcga-pancan          |           |   302
 #   Gisette              |           |   189



datasets = {
            'leukemia',                                                                          
            'gcm',                                                                               
            'madelon',                                                                           
            'gisette',                                                                           
            'all-leukemia', 
            'arrhythmia',
            'dermatology',
            'isolet',
            'mfeat',
            'movement',
            'optdigits',
            'smartphone-activity',
            'tcga-pancan',
            'yeoh'
            }


fsmethods = {
             #'sk_ReliefF', 
             #'sk_SURF', 
             #'sk_SURFstar', 
             #'sk_MultiSURF', 
             #'sk_MultiSURFstar', 
             ##'sk_TuRF', 
             #'py_ReliefF', 
             #'MutualInfo', 
             #'RFE-DT', 
             #'KBest-Ftest', 
             #'SFS-DT', 
             'SFFS-DT', 
             #'SBS-DT', 
             'SBFS-DT'
             }

experiment_id = 'Floating_Search_Wrappers'
#config_path = 'settings/settingsDT/'
config_path = 'settings/'

# Numer of Neighbors for Relief based methods
_nb = 20


try:
    
    sheets = []
    writer = pd.ExcelWriter('reporte_'+experiment_id+'_nb'+str(_nb)+'.xls', engine='xlsxwriter') 
    workbook  = writer.book

    # path='other_methods/python/'
    path='../../'                                                                               
                                                                                                
                                                                                                
    dtset = 'leukemia'         
    if dtset in datasets:
        file1 = path+'/data/leukemia_train_38x7129.arff'                                            
        file2 = path+'/data/leukemia_test_34x7129.arff'                                             
        nfeats = 30                                                                                
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)                    
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)                              
                                                                                                
                                                                                                
    dtset = 'gcm'                                                                               
    if dtset in datasets:
        file1 = path+'/data/GCM_Training.arff'                                                      
        file2 = path+'/data/GCM_Test.arff'                                                          
        nfeats = 264                                                                               
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)

                                                                                                
    dtset = 'madelon'                                                                           
    if dtset in datasets:
        file1 = path+'/data/madelon.trn.arff'                                                       
        file2 = path+'/data/madelon.tst.arff'                                                       
        nfeats = 22                                                                                
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)                              
                                                                                                
                                                                                                
    dtset = 'gisette'                                                                           
    if dtset in datasets:
        file1 = path+'/data/Gisette/gisette_train.arff'                                             
        file2 = path+'/data/Gisette/gisette_test.arff'                                              
        nfeats = 189                                                                                
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset ='all-leukemia' 
    if dtset in datasets:
        file1 = path+'/data/additional/ALL-Leukemia_trn.arff'
        file2 = path+'/data/additional/ALL-Leukemia_tst.arff'
        nfeats = 168
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'arrhythmia'
    if dtset in datasets:
        file1 = path+'/data/additional/arrhythmia_trn.arff'
        file2 = path+'/data/additional/arrhythmia_tst.arff'
        nfeats = 11
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'dermatology'
    if dtset in datasets:
        file1 = path+'/data/additional/dermatology-5dobscv_trn.arff'
        file2 = path+'/data/additional/dermatology-5dobscv_tst.arff'
        nfeats = 6
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'isolet'
    if dtset in datasets:
        file1 = path+'/data/additional/isolet_trn.arff'
        file2 = path+'/data/additional/isolet_tst.arff'
        nfeats = 44
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'mfeat'
    if dtset in datasets:
        file1 = path+'/data/additional/mfeat_trn.arff'
        file2 = path+'/data/additional/mfeat_tst.arff'
        nfeats = 14
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'movement'
    if dtset in datasets:
        file1 = path+'/data/additional/movement_libras-5dobscv_trn.arff'
        file2 = path+'/data/additional/movement_libras-5dobscv_tst.arff'
        nfeats = 9
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'optdigits'
    if dtset in datasets:
        file1 = path+'/data/additional/optdigits-5dobscv_trn.arff'
        file2 = path+'/data/additional/optdigits-5dobscv_tst.arff'
        nfeats = 10
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'smartphone-activity'
    if dtset in datasets:
        file1 = path+'/data/additional/smartphone_activity_trn.arff'
        file2 = path+'/data/additional/smartphone_activity_tst.arff'
        nfeats = 8
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'tcga-pancan'
    if dtset in datasets:
        file1 = path+'/data/additional/tcga-pancan_trn.arff'
        file2 = path+'/data/additional/tcga-pancan_tst.arff'
        nfeats = 302
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    dtset = 'yeoh'
    if dtset in datasets:
        file1 = path+'/data/additional/yeoh-trn.arff'
        file2 = path+'/data/additional/yeoh-tst.arff'
        nfeats = 270
        tmpres = batchRelief2(file1, nfeats, fsmethods, nb=_nb, mpath=path, dataset=dtset, tag=experiment_id, confpath=config_path)
        sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


    writer.save()
    notification.notify("La instancia de fsmethods " + experiment_id + " ha finalizado exitosamente.")
    
except:
    notification.notify("Han surgido errores en la instancia de fsmethods " + experiment_id + ".")
    raise


    
