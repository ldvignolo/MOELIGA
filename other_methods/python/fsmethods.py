# pip3 install skrebate    
# pip3 install ReliefF
# pip3 install arff


import json
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TuRF
from ReliefF import ReliefF as ReliefF2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
#from sklearn.feature_selection import SequentialFeatureSelector, f_classif
from sklearn.feature_selection import SelectKBest, f_classif

import os

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score

import time


## 4 Árbol 
def clf_TREE(X_train, X_test, y_train, y_test):
    dtree = DecisionTreeClassifier(random_state=0, max_depth=2)
    dtree.fit(X_train, y_train) 
    y_pred = dtree.predict(X_test) 
    acc = accuracy_score(y_test, y_pred)
    uar = balanced_accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None)[1]
    nf = X_train.shape[1]    
    return acc, uar, rec, nf




## 5 RandomForest 
def clf_RF(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    uar = balanced_accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None)[1]
    nf = X_train.shape[1]    
    return acc, uar, rec, nf




## 6 SVM
def clf_SVM(X_train, X_test, y_train, y_test):
    #clf = svm.SVC(kernel='linear') # Linear Kernel
    clf = svm.SVC(kernel='poly', degree=1, coef0=0, gamma='auto') # polynomial Kernel
    #clf = svm.SVC(kernel='rbf',gamma=1) # RBF Kernel
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    uar = balanced_accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None)[1]
    nf = X_train.shape[1]    
    return acc, uar, rec, nf



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


def batchRelief(file1, file2, nf, encodeLabels=True, nb=20):
    
    print('\n < '+dataset+' >\n')
    
    trnData, tstData, trnLabels, tstLabels = loadDataset(file1,file2,encodeLabels)
    trnData, tstData = estandarizar(trnData, tstData)

    num_examples = len(trnLabels)
    num_features = trnData.shape[0]

    if num_examples<nb:
        nb = num_examples-1

    uar1 = 0.0
    uar2 = 0.0
    uar3 = 0.0
    uar4 = 0.0
    uar5 = 0.0
    uar6 = 0.0

    fs = ReliefF(n_features_to_select=nf, n_neighbors=nb)
    fs.fit(trnData, trnLabels)

    X_trn = fs.transform(trnData)
    X_tst = fs.transform(tstData) 
   
    _, uar1, _, _ = clf_TREE(X_trn, X_tst, trnLabels, tstLabels)
    _, uar2, _, _ = clf_RF(X_trn, X_tst, trnLabels, tstLabels)
    _, uar3, _, _ = clf_SVM(X_trn, X_tst, trnLabels, tstLabels)

    fs = ReliefF2(n_neighbors=nb, n_features_to_keep=nf)
    fs.fit(trnData, trnLabels)

    X_trn = fs.transform(trnData)
    X_tst = fs.transform(tstData)   

    _, uar4, _, _ = clf_TREE(X_trn, X_tst, trnLabels, tstLabels)    
    _, uar5, _, _ = clf_RF(X_trn, X_tst, trnLabels, tstLabels)    
    _, uar6, _, _ = clf_SVM(X_trn, X_tst, trnLabels, tstLabels)

    print('      skrebate  /  ReliefF\n')
    print('DT:     %2.2f '% uar1,  '  /   %2.2f'% uar4 )
    print('RF:     %2.2f '% uar2,  '  /   %2.2f'% uar5 )
    print('SVM:    %2.2f '% uar3,  '  /   %2.2f'% uar6 )
    

def batchRelief2(file1, nf, encodeLabels=True, nb=20, mpath='None', dataset='None'):
    
    if nf==0:
        return
    
    print('\n < '+dataset+' >')

    testbin  = 'bin/test'
    confpath = 'settings/settingsDT/'
    prevpath=os.getcwd()
    results = {}
    
    if not os.path.exists(prevpath + '/_resultados/'):
        os.makedirs(prevpath + '/_resultados/')
  
    trnData, trnLabels, header = loadDataset(file1, encodeLabels)
    trnData = estandarizar(trnData)    

    num_examples = len(trnLabels)
    num_features = trnData.shape[0]

    if num_examples<nb:
        nb = num_examples-1

    fsmethod = 'sk_ReliefF'    
    start_time = time.time()
    fs = ReliefF(n_features_to_select=nf, n_neighbors=nb)
    fs.fit(trnData, trnLabels)
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = fs.top_features_[:nf]
    fv.sort()
    fv = [x+1 for x in fv] # 0 based 
    # featsfile = mpath + '_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_SURF'
    start_time = time.time()
    fs = SURF(n_features_to_select=nf)
    fs.fit(trnData, trnLabels)
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = fs.top_features_[:nf]
    fv.sort()
    fv = [x+1 for x in fv] # 0 based
    # featsfile = mpath + '_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_SURFstar'
    start_time = time.time()
    fs = SURFstar(n_features_to_select=nf)
    fs.fit(trnData, trnLabels)
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = fs.top_features_[:nf]
    fv.sort()
    fv = [x+1 for x in fv] # 0 based
    # featsfile = mpath + '_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_MultiSURF'
    start_time = time.time()
    fs = MultiSURF(n_features_to_select=nf)
    fs.fit(trnData, trnLabels)
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = fs.top_features_[:nf]
    fv.sort()
    fv = [x+1 for x in fv] # 0 based
    # featsfile = mpath + '_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'sk_MultiSURFstar'
    start_time = time.time()
    fs = MultiSURFstar(n_features_to_select=nf)
    fs.fit(trnData, trnLabels)
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = fs.top_features_[:nf]
    fv.sort()
    fv = [x+1 for x in fv] # 0 based
    # featsfile = mpath + '_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})
    
    # # #
    
    #fsmethod = 'sk_TuRF'
    #start_time = time.time()
    ##fs = TuRF(core_algorithm="ReliefF", n_features_to_select=nf, pct=0.5)
    #fs = TuRF(core_algorithm="ReliefF", n_features_to_select=nf)
    #fs.fit(trnData, trnLabels, fit_params={'headers': header})
    #elapsed_time = time.time() - start_time
    #etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    #fv = fs.top_features_[:nf]
    #fv.sort()
    #fv = [x+1 for x in fv] # 0 based
    ## featsfile = mpath + '_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    #featsfile = prevpath + '/_resultados/' + dataset+'_features_' + fsmethod + '.txt'
    #f=open(featsfile,'w')
    #for ele in fv:
        #f.write(str(ele)+' ')
    #f.write('\n')    
    #f.close()

    ##jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    #jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    #cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    #os.chdir(mpath)
    #os.system(cmd)
    #os.chdir(prevpath)
    #tmpdict = loadJSON(jsonfile,etime,nf,nb)
    #results.update({fsmethod:tmpdict})
    
    # # #

    fsmethod = 'py_ReliefF'
    start_time = time.time()
    fs = ReliefF2(n_neighbors=nb, n_features_to_keep=nf)
    fs.fit(trnData, trnLabels)
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = fs.top_features[:nf]
    fv.sort()
    fv = [x+1 for x in fv] # 0 based
    #featsfile = mpath + '_resultados/' + dataset+'_features_'+ fsmethod +'.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_'+ fsmethod +'.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})
    
    # # #
    
    fsmethod = 'MutualInfo'
    start_time = time.time()
    mi_features = mutual_info_classif(trnData, trnLabels, random_state=42)
    fv = np.argsort(mi_features)[-1:0:-1][:number_of_features]
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = fv[:nf]
    fv.sort()
    fv = [x+1 for x in fv] # 0 based
    #featsfile = mpath + '_resultados/' + dataset+'_features_'+ fsmethod +'.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_'+ fsmethod +'.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})
    
    # # # 
    
    fsmethod = 'KBest-DT'
    #fsmethod = 'SFS-DT'
    start_time = time.time()
    dtree = DecisionTreeClassifier(random_state=0, max_depth=2)    
    sfs = SelectKBest(dtree, n_features_to_select=nf)
    #sfs = SequentialFeatureSelector(dtree, n_features_to_select=nf)
    sfs.fit(X, y)
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = sfs.get_support(indices=True)
    fv.sort()
    fv = [x+1 for x in fv] # 0 based
    #featsfile = mpath + '_resultados/' + dataset+'_features_'+ fsmethod +'.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_'+ fsmethod +'.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})
    
    # # # 
    # Univariate feature selection with F-test for feature scoring
    fsmethod = 'KBest-Ftest'
    #fsmethod = 'SFS-Ftest'
    start_time = time.time()
    sfs = SelectKBest(f_classif, n_features_to_select=nf)
    #sfs = SequentialFeatureSelector(f_classif, n_features_to_select=nf)
    sfs.fit(X, y)
    elapsed_time = time.time() - start_time
    etime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    fv = sfs.get_support(indices=True)
    fv.sort()
    fv = [x+1 for x in fv] # 0 based
    #featsfile = mpath + '_resultados/' + dataset+'_features_'+ fsmethod +'.txt'
    featsfile = prevpath + '/_resultados/' + dataset+'_features_'+ fsmethod +'.txt'
    f=open(featsfile,'w')
    for ele in fv:
        f.write(str(ele)+' ')
    f.write('\n')    
    f.close()

    #jsonfile = mpath + '_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    jsonfile = prevpath + '/_resultados/' + dataset + '_' + fsmethod + '_mlpack.test'
    cmd = testbin + ' file ' + featsfile + ' cfg ' + confpath + dataset + '_SETTINGS.cfg' + ' > ' + jsonfile
    os.chdir(mpath)
    os.system(cmd)
    os.chdir(prevpath)
    tmpdict = loadJSON(jsonfile,etime,nf,nb)
    results.update({fsmethod:tmpdict})    
    
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


_nb = 20

sheets = []
writer = pd.ExcelWriter('reporte_nb'+str(_nb)+'.xls', engine='xlsxwriter') 
workbook  = writer.book

# path='other_methods/python/'
path='../../'                                                                                 #               MOELIGA - NFEATS
                                                                                              #
                                                                                              #                        |  2nd opt  | 1rst opt
dtset = 'leukemia'                                                                            #   dermatology          |     5     |     6
file1 = path+'/data/leukemia_train_38x7129.arff'                                              #   optdigits            |           |     
file2 = path+'/data/leukemia_test_34x7129.arff'                                               #   movement             |     8     |     9
nfeats = 30                                                                                   #   arrhythmia           |    12     |    11
tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)                       #   madelon              |    24     |    22
sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)                                #   smartphone-activity  |           |    
                                                                                              #   isolet               |    37     |    44
                                                                                              #   mfeat                |    17     |    14
#dtset = 'gcm'                                                                                 #   leukemia             |    30     |    30
#file1 = path+'/data/GCM_Training.arff'                                                        #   all-leukemia         |   145     |   168
#file2 = path+'/data/GCM_Test.arff'                                                            #   yeoh                 |   298     |   270
#nfeats = 264                                                                                  #   gcm                  |   241     |   264
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)                       #   tcga-pancan          |           |
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'madelon'
#file1 = path+'/data/madelon.trn.arff'
#file2 = path+'/data/madelon.tst.arff'
#nfeats = 22
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'gisette'
#file1 = path+'/data/Gisette/gisette_train.arff'
#file2 = path+'/data/Gisette/gisette_test.arff'
#nfeats = 0
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset ='all-leukemia' 
#file1 = path+'/data/additional/ALL-Leukemia_trn.arff'
#file2 = path+'/data/additional/ALL-Leukemia_tst.arff'
#nfeats = 168
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'arrhythmia'
#file1 = path+'/data/additional/arrhythmia_trn.arff'
#file2 = path+'/data/additional/arrhythmia_tst.arff'
#nfeats = 11
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'dermatology'
#file1 = path+'/data/additional/dermatology-5dobscv_trn.arff'
#file2 = path+'/data/additional/dermatology-5dobscv_tst.arff'
#nfeats = 6
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'isolet'
#file1 = path+'/data/additional/isolet_trn.arff'
#file2 = path+'/data/additional/isolet_tst.arff'
#nfeats = 44
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'mfeat'
#file1 = path+'/data/additional/mfeat_trn.arff'
#file2 = path+'/data/additional/mfeat_tst.arff'
#nfeats = 14
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'movement'
#file1 = path+'/data/additional/movement_libras-5dobscv_trn.arff'
#file2 = path+'/data/additional/movement_libras-5dobscv_tst.arff'
#nfeats = 9
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'optdigits'
#file1 = path+'/data/additional/optdigits-5dobscv_trn.arff'
#file2 = path+'/data/additional/optdigits-5dobscv_tst.arff'
#nfeats = 0
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'smartphone-activity'
#file1 = path+'/data/additional/smartphone_activity_trn.arff'
#file2 = path+'/data/additional/smartphone_activity_tst.arff'
#nfeats = 0
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'tcga-pancan'
#file1 = path+'/data/additional/tcga-pancan_trn.arff'
#file2 = path+'/data/additional/tcga-pancan_tst.arff'
#nfeats = 0
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


#dtset = 'yeoh'
#file1 = path+'/data/additional/yeoh-trn.arff'
#file2 = path+'/data/additional/yeoh-tst.arff'
#nfeats = 270
#tmpres = batchRelief2(file1, nfeats, nb=_nb, mpath=path, dataset=dtset)
#sheets, writer = addsheet(dtset,tmpres,sheets,writer,workbook)


writer.save()
