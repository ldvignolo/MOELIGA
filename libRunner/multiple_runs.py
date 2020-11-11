import os
import glob
import numpy as np
import json
import yaml
#import streamlit as st
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import scipy.stats as st
from scipy.stats import median_absolute_deviation as mad
from scipy.stats import mode as mode

import inspect

import pandas as pd
from tqdm import tqdm
import io

from multiprocessing import Pool

#===============================================
def check_type(values):
    '''
    '''
    
    if isinstance(values, np.ndarray):
        values = values.tolist()
    
    elif isinstance(values, np.float64):
        values = float(values)
    
    elif isinstance(values, np.int64):
        values = int(values)
    
    return values
#===============================================


#====================================================
def calculate_confidence_interval(data, alpha=0.05):
    '''
    REFERENCIA: https://rpubs.com/acatania/396921
                https://stackoverrun.com/es/q/5277580
    
    Walpole, Myers, Myesr, 'Probabilidad y Estadística para ingenieros'
    6 Ed.
    [8 ed]--> https://drive.google.com/file/d/0B9FlUGP8i39reUg0Qk9NWDlkX2c/view
    '''
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    N = data.shape[0]
    
    if (N < 30):
        
        CI = st.t.ppf(1-(alpha/2), N-1) * np.std(data) / np.sqrt(N) # pp 247
    
    else:
        
        CI = st.norm.ppf(1-(alpha/2)) * np.std(data) / np.sqrt(N) # pp 244
    
    if np.isnan(CI):
        CI = 0.0
    
    return CI
#====================================================


#====================================================
def statistics(values, statistic, axis=0, alpha=0.05):
    '''
    '''
    
    if (values != None):
        
        if isinstance(values, list):
            values = np.array(values)
        
        if isinstance(values, np.ndarray) and (values.ndim == 1):
            axis = 0
        
        if isinstance(values, np.ndarray):
            
            #---------------------------------------
            if (statistic == 'mean'):
                values = np.mean(values, axis=axis)
            
            elif (statistic == 'median'):
                values = np.median(values, axis=axis)
            
            elif (statistic == 'std'):
                values = np.std(values, axis=axis)
            
            elif (statistic == 'mad'):
                values = mad(values, axis=axis)
                
                if isinstance(values,tuple):
                    values = values[0].flatten()
            
            elif (statistic == 'max'):
                values = np.max(values, axis=axis)
            
            elif (statistic == 'min'):
                values = np.min(values, axis=axis)
            
            elif (statistic == 'mode'):
                pass
            
            elif (statistic == 'sum'):
                values = np.sum(values, axis=axis)
            
            elif (statistic == 'confidence'):
                values = calculate_confidence_interval(values, alpha=alpha)
        #---------------------------------------
        
        values = check_type(values)
    
    return values
#====================================================


#====================================================
def apply_statistic(values, statistic=None, squeezy_criterium='mean'):
    '''
    '''
    
    if (values != None):
        
        # COMPRIME LOS VALORES DE LA GENERACION
        values = statistics(values,
                            statistic=squeezy_criterium,
                            axis=1)
        
        # CALCULA ESTADISTICA
        values = statistics(values,
                            statistic=statistic,
                            axis=0)  # axis=1
        
    
    return values
#====================================================



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def procesar_replica(info):
    '''
    PROCESO REPLICAS
    '''
    
    path, lib_path = info
    
    print('\n[{}/{}] Procesando {}...'.format(len(paths), path))
    
    mr = MULTIPLE_RUNS(path, lib_path=lib_path)
    
    mr.build_report(show=False, save=True)
    
    mr.plot_report()
    
    del mr
    
    print('Done!!\n')
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def procesar_replicas(root_path, lib_path, n_jobs=2):
    '''
    Esta funcion recorre todas las carpetas desde el root del experimento
    completo y procesa cada corrida generando las gráficas y el summary de
    cada experimento ("experiment_summart.json").
    
    '''
    
    if not isinstance(n_jobs, int):
        n_jobs = int(n_jobs)
    
    paths_to_subfolders = [x[0] for x in os.walk('{}'.format(root_path))]
    
    paths = []
    
    #-----------------------------------
    # EXTRAIGO PATHS DE LAS REPLICAS
    #-----------------------------------
    for path in tqdm(paths_to_subfolders):
        
        filename = glob.glob(os.path.join(path,'experiment_summary.json'))  # Devuelve una lista
        
        if filename:
            
            path,_ = os.path.split(path)
            if (path not in paths):
                paths.append(path)
    
    
    
    with Pool(n_jobs) as p:
        p.map(procesar_replica, zip(paths,[lib_path]*len(paths)))
    
    ##-----------------------------------
    ## PROCESO REPLICAS
    ##-----------------------------------
    #for n,path in enumerate(paths):
        
        #print('\n[{}/{}] Procesando {}...'.format(n+1, len(paths), path))
        
        #mr = MULTIPLE_RUNS(path, lib_path=lib_path)
        
        #mr.build_report(show=False, save=True)
        
        #mr.plot_report()
        
        #del mr
        
        #print('Done!!\n')
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class CLASIFIER(object):
    '''
    '''
    
    #===============================================
    def __init__(self, name):
        '''
        '''
        
        self.name = name
        self.metrics = dict()
        #self.time = []
    
    
    #===============================================
    def squeezing(self, values, criterium=None, axis=0):
        
        new_values = []
        
        for value in values:
            new_values.append(statistics(value, statistic=criterium, axis=axis))
        
        return new_values
    
    
    #===============================================
    def get_statistic(self, values, statistic=None, squeezy_criterium=None, axis=0):
        
        values = self.squeezing(values,
                                criterium=squeezy_criterium,
                                axis=axis)
        
        values = statistics(values,
                           statistic=statistic,
                           axis=0)
        
        return values

    
    #===============================================
    def update_metric(self, metric_name, values):
        '''
        '''
        
        if (metric_name not in self.metrics.keys()):
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(values)
    
    
    #===============================================
    def get_metric(self, metric, statistic='mean', squeezy_criterium=None):
        '''
        '''
        
        values = self.metrics.get(metric)
        
        values = self.get_statistic(values=values,
                                    statistic=statistic,
                                    squeezy_criterium=squeezy_criterium)
        
        return values
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        
        


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class CLASIFIERS(dict):
    '''
    '''
    
    #================================================
    def __init__(self, classifiers, squeezy_criterium='mean'):
        
        self.update(classifiers)
    
        
    #================================================
    def get_metric(self, classifier, metric, statistic, squeezy_criterium):
        '''
        '''
        values = self[classifier].get_metric(metric, statistic, squeezy_criterium)
        
        return values
    
    
    #================================================
    def update_metrics(self, classifier, metrics):
        
        for k,v in metrics.items():
            
            self[classifier].update_metric(k, v)
    
    
    #================================================
    def update(self, classifiers):
        '''
        '''
        
        if isinstance(classifiers, dict):
            
            # PARA CADA CLASIFICADOR DISPONIBLE
            for classifier, data in classifiers.items():
                
                if classifier not in self.keys():
                    self[classifier] = CLASIFIER(classifier)
                
                self.update_metrics(classifier, data)
        
        else:
            print('Debe proveer un diccionario para crear el objeto.')
    
    
    #================================================
    def get_classifiers(self):
        '''
        '''
        return self.keys()
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class MEASURES(dict):
    '''
    '''
    
    #================================================
    def __init__(self, measures):
        
        self.update_measures(measures)
    
    
    #================================================
    def update_measures(self, measures):
        '''
        '''
        
        if isinstance(measures, dict):
            
            for k,v in measures.items():
                
                if (k not in self):
                    self[k] = []
                
                self[k].append(v)
    
    
    #================================================
    def available_measures(self):
        '''
        '''
        return list(self.keys())
    
    
    #===============================================
    def get_metric(self, metric, statistic='mean', squeezy_criterium='mean'):
        '''
        '''
        
        values = self[metric.upper()]
        
        values = apply_statistic(values=values,
                                 statistic=statistic,
                                 squeezy_criterium=squeezy_criterium)
        
        return values
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class MULTIPLE_RUNS(object):
    '''
    Extrae la información para la repeticiones de un experimento y la ordena para acceder de forma sencilla.
    '''
    
    #====================================================
    def __init__(self, path, lib_path):
        '''
        '''
        
        self.lib_path = lib_path
        
        #-----------------------------------------------------------
        with open(self.lib_path + 'repetitions_summary_template.yaml', 'r') as fp:
            TEMPLATE = yaml.load(fp, Loader=yaml.FullLoader)
        #-----------------------------------------------------------
        
        self.train = dict()
        self.test = dict()
        
        self.classifiers = set()
        
        #self.repetitions = {'TRAIN': dict(),
                            #'TEST': dict()}
        
        self.path = path
        self.fullpath = os.path.abspath(path)
                
        self.evaluated_parameters = self._get_evaluated_parameters()
        
        # EXTRAIGO NOMBRE DE LAS CARPETAS DONDE SE GUARDARON LAS REPETICIONES
        subfolders = [x[1] for x in os.walk('{}'.format(self.path))][0]
        
        # RECORRO CADA REPETICION
        for subfolder in subfolders:
            
            fullname = os.path.join(self.path, subfolder, 'experiment_summary.json')
            
            with open(fullname, 'r') as fp:
                data = json.load(fp)
            
            
            #==========================
            # ALMACENO DATOS TRAIN
            #==========================
            if (len(self.train) == 0):
                
                self.train = MEASURES(data['TRAIN'])
                
            else:
                
                self.train.update_measures(data['TRAIN'])
                
            
            
            #==========================
            # ALMACENO DATOS TEST
            #==========================
            for criterium in data['TEST'].keys():
                
                if criterium not in self.test.keys():
                    
                    self.classifiers.update(list(data['TEST'][criterium]['CLASIFICADORES'].keys()))
                    
                    self.test[criterium] = CLASIFIERS(data['TEST'][criterium]['CLASIFICADORES'])
                    self.test[criterium]['MEASURES'] = dict()
                    
                    for key in data['TEST'][criterium].keys():
                        
                        if (key != 'CLASIFICADORES'):
                            
                            self.test[criterium]['MEASURES'][key] = [data['TEST'][criterium][key]]
                    
                
                else:
                    
                    for key in data['TEST'][criterium].keys():
                        
                        if (key == 'CLASIFICADORES'):
                            self.test[criterium].update(data['TEST'][criterium][key])
                        
                        else:
                            self.test[criterium]['MEASURES'][key].append(data['TEST'][criterium][key])
                    
            
            self.classifiers = list(self.classifiers)
            self.classifiers.sort()
            
    #===============================================================================
    
    
    
    #========================================================
    def _get_evaluated_parameters(self):
        '''
        '''
        
        path = self.path.strip('/')  # Elimina el simbolo inicial
        
        parameters = []
        
        while (path != ''):
            
            path,folder = os.path.split(path)
            
            if ('__' in folder):
                
                parameter,value = folder.split('__')
                parameters.append([parameter,value])
        
        return parameters
    #========================================================
    
    
    #====================================================
    def plot_measure_evolution(self, measure, measure_name, ax=None, show=True, save=False):
        '''
        '''
        
        if (ax == None):
            fig, ax = plt.subplots(1, 1, figsize=(8,2))
        
        #-----------------
        # DATA
        #-----------------
        l1 = ax.plot(measure['G'], measure['mean'],'-r', linewidth=2)
        l2 = ax.plot(measure['G'], measure['median'],'-b', linewidth=1, alpha=0.5)
        
        ax.fill_between(measure['G'],
                        measure['max'],
                        measure['min'],
                        facecolor='yellow', alpha=0.5)
        
        
        ax.set_title(u'Evolution of {}'.format(measure_name), fontsize=7)
        ax.set_xlabel(u'Generations', fontsize=7)
        ax.set_ylabel(u'Measure', fontsize=7)
        
        ax.grid(True)
        ax.tick_params(axis='x', which='major', labelsize=7)
        ax.tick_params(axis='y', which='major', labelsize=7)
        
        ax.legend([r'mean', r'median'], loc='best')
        
        plt.tight_layout()
        
        
        if save:
            plt.savefig(os.path.join(self.path,'evolution_of_{}.pdf'.format(measure_name)), dpi=600)
            plt.savefig(os.path.join(self.path,'evolution_of_{}.png'.format(measure_name)), dpi=600)
        
        if show:
            plt.show()
        
        if (ax == None):
            plt.close(fig)
    #====================================================
    
    
    #====================================================
    def plot_summary_train(self, measures=[], show=True, save=False):
        '''
        '''
        
        N = len(measures)
        
        #--------------------------------------------------------------------
        fig, ax = plt.subplots(N, 1, figsize=(8, 2*N))
        
        for idx, measure_name in enumerate(measures):
            
            data = []
            
            transform = False
            for i,rep in enumerate(self.train[measure_name]):  # rep no está vacío
                                                               # y es una lista
                
                if (rep) and isinstance(rep[0], list):
                    
                #-----------------------------
                    for j,d in enumerate(rep):
                        if (i == 0):
                            data.append(d)
                        else:
                            data[j].extend(d)
                    
                #-----------------------------
                else:
                    data.append(rep)
                    transform = True
                    
                #-----------------------------
            
            if transform:
                data = np.array(data)
                data = data.T.tolist()
            
            measure = {'mean': np.zeros(len(data)),
                       'median': np.zeros(len(data)),
                       'max': np.zeros(len(data)),
                       'min': np.zeros(len(data)),
                       'G': None}
            
            for i,d in enumerate(data):
                
                d = np.array(d).astype(np.float64)
                
                measure['mean'][i] = np.mean(d)
                measure['median'][i] = np.median(d)
                measure['max'][i] = np.max(d)
                measure['min'][i] = np.min(d)
            
            measure['G'] = np.arange(len(data))
            
            if (N > 1):
                self.plot_measure_evolution(measure, measure_name=measure_name, ax=ax[idx], show=False, save=False)
            else:
                self.plot_measure_evolution(measure, measure_name=measure_name, ax=ax, show=False, save=False)
            
        
        #--------------------------------------------------------------------
        
        plt.tight_layout()
        
        
        if save:
            plt.savefig(os.path.join(self.path,'evolution_of_several_measures.pdf'), dpi=300)
            plt.savefig(os.path.join(self.path,'evolution_of_several_measures.png'), dpi=300)
        
        if show:
            plt.show()
    
        
        plt.close(fig)  # NO OLVIDAR ESTO PARA QUE NO QUEDE CARGADA EN MEMORIA!!!
                        # https://heitorpb.github.io/bla/2020/03/18/close-matplotlib-figures/
    #====================================================
    
    #====================================================
    def plot_summary_test(self, measures=[], show=True, save=False):
        '''
        '''
        
        CRITERIOS = list(self.test.keys())
        
        N = len(measures) * len(CRITERIOS) * len(self.classifiers)
        
        #--------------------------------------------------------------------
        fig, ax = plt.subplots(N, 1, figsize=(8, 2*N))
        
        for criterio in CRITERIOS:
            
            for classifier in self.classifiers:
            
                for idx, measure_name in enumerate(measures):
                    
                    data = np.array(self.test[criterio][classifier].metrics[measure_name])
                    
                    MEASURE = {'mean':[], 'median':[], 'max':[], 'min':[], 'G': None}
                    
                    #-------------------------------------------------------------
                    if (data.ndim > 2):
                        data = np.concatenate(data, axis=1)
                        data = data.T
                    #-------------------------------------------------------------
                    
                    MEASURE['mean'] = np.mean(data, axis=0)
                    MEASURE['median'] = np.median(data, axis=0)
                    MEASURE['max'] = np.max(data, axis=0)
                    MEASURE['min'] = np.min(data, axis=0)
                    MEASURE['G'] = np.arange(data.shape[1])
                    
                    self.plot_measure_evolution(MEASURE, measure_name=measure_name, ax=ax[idx], show=False, save=False)
            
        
        #--------------------------------------------------------------------
        
        plt.tight_layout()
        
        
        if save:
            plt.savefig(os.path.join(self.path,'evolution_of_several_measures.pdf'), dpi=300)
            plt.savefig(os.path.join(self.path,'evolution_of_several_measures.png'), dpi=300)
        
        if show:
            plt.show()
    
        
        plt.close(fig)  # NO OLVIDAR ESTO PARA QUE NO QUEDE CARGADA EN MEMORIA!!!
                        # https://heitorpb.github.io/bla/2020/03/18/close-matplotlib-figures/
    #====================================================
    
    
    #====================================================
    def plot_objectives_by_criterium(self, criterium=['R1', 'R2'], show=True, save=False):
        '''
        Esta función grafica los objetivos del fitness para los "criterios" especificados.
        El gráfico se construye para seguir la evolución del fitness de los individuos
        del frente de pareto, de acuerdo al criterio elegido.
        '''
        
        _KEYS = [key for key in self.train.keys() if ('_OBJETIVOS' in key)]  # Claves guardadas en "train"
        
        KEYS = []
        for criterio in criterium:
            
            key = criterio + '_OBJETIVOS'
            
            if key in _KEYS:
                
                KEYS.append(key)
        
        #--------------------------------------------------------------------
        X = np.array(self.train[KEYS[0]], dtype=np.float)
        
        N = X.shape[-1]
        
        fig, ax = plt.subplots(N, len(KEYS), figsize=(5*len(KEYS), 2*N))
        #--------------------------------------------------------------------
        
        for i,key in enumerate(KEYS):
            
            X = np.array(self.train[key], dtype=np.float)
            
            for j in range(X.shape[-1]):
                
                MEASURE = dict()
                
                MEASURE['mean'] = np.mean(X[:,:,j], axis=0)
                MEASURE['median'] = np.median(X[:,:,j], axis=0)
                MEASURE['max'] = np.max(X[:,:,j], axis=0)
                MEASURE['min'] = np.min(X[:,:,j], axis=0)
                
                MEASURE['G'] = np.arange(X.shape[1])
                
                measure_name = 'OBJETIVO {} [{}]'.format(j,key.replace('_OBJETIVOS',''))
                
                self.plot_measure_evolution(MEASURE, measure_name=measure_name, ax=ax[j,i], show=False, save=False)
        
        
        plt.tight_layout()
        
        
        if save:
            plt.savefig(os.path.join(self.path,'evolution_of_several_measures_for_pareto.pdf'), dpi=300)
            plt.savefig(os.path.join(self.path,'evolution_of_several_measures_for_pareto.png'), dpi=300)
        
        if show:
            plt.show()
    
        
        plt.close(fig)  # NO OLVIDAR ESTO PARA QUE NO QUEDE CARGADA EN MEMORIA!!!
                        # https://heitorpb.github.io/bla/2020/03/18/close-matplotlib-figures/
    
    #====================================================
    
    
    #@@@@@@@@@@@@@@@@@@@@@@@
    # REPORT
    #@@@@@@@@@@@@@@@@@@@@@@@
    
    #====================================================
    def build_report(self, show=False, save=True):
        '''
        '''
        
        #-----------------------------------------------------------
        with open(self.lib_path + 'repetitions_summary_template.yaml', 'r') as fp:
            TEMPLATE = yaml.load(fp, Loader=yaml.FullLoader)
        #-----------------------------------------------------------
        
        report = ''
        
        if self.evaluated_parameters:
            
            for (parameter,value) in self.evaluated_parameters[::-1]:
                report += ',{}={}\n'.format(parameter, value)
        
        
        #================
        # TRAIN
        #================
        report += 'TRAIN,\n'
        
        
        #----------
        # TIEMPO
        #----------
        if (TEMPLATE['TRAIN'] != None):
            
            for measure in TEMPLATE['TRAIN'].keys():
                
                squeezy_criterium = TEMPLATE['TRAIN'][measure]['SQUEEZE_CRITERIUM']
                
                for statistic in TEMPLATE['TRAIN'][measure]['STATISTICS']:
                    
                    if (measure in self.train.available_measures()):
                        
                        values = self.train.get_metric(measure,
                                                    statistic=statistic,
                                                    squeezy_criterium=squeezy_criterium)
                        
                        
                        report += '{}_{},{}\n'.format(measure,
                                                    statistic,
                                                    values
                                                    )
                                              
        
        
        #================
        # TEST
        #================
        report += 'TEST,\n'
        
        for criterium in self.test.keys():
            
            report += 'CRITERIO_{},\n'.format(criterium)
            
            for measure in self.test[criterium]['MEASURES'].keys():
                
                #---------------
                # MEASURES
                #---------------
                if measure in TEMPLATE['TEST']['MEASURES'].keys():
                    
                    # CRITERIO DE SQUEEZING POR GENERACION
                    squeezy_criterium = TEMPLATE['TEST']['MEASURES'][measure]['SQUEEZE_CRITERIUM']
                    
                    # APLICO ESTADÏSTICAS
                    for statistic in TEMPLATE['TEST']['MEASURES'][measure]['STATISTICS']:
                        
                        values = statistics(self.test[criterium]['MEASURES'][measure],
                                            statistic=statistic)
                        
                        if isinstance(values, list):
                            values = values[0]
                        
                        report += '{}_{},{}\n'.format(measure,statistic,values)
                
            
            #-------------
            # METRICS
            #-------------
            classifiers = list(self.test[criterium].keys())
            classifiers.remove('MEASURES')
            
            for classifier in classifiers:
                
                report += '{},\n'.format(classifier)
                
                for metric in TEMPLATE['TEST']['METRICS'].keys():
                    
                    squeezy_criterium = TEMPLATE['TEST']['METRICS'][metric]['SQUEEZE_CRITERIUM']
                    
                    for statistic in TEMPLATE['TEST']['METRICS'][metric]['STATISTICS']:
                        
                        #----------------------------------------------------------------------------
                        # SE EXTRAE LA METRICA SIN APLICAR LA ESTADISTICA (SOLO SQUEEZE_CRITERIUM)
                        #----------------------------------------------------------------------------
                        values = self.test[criterium].get_metric(classifier,
                                                                 metric,
                                                                 statistic=None,
                                                                 squeezy_criterium=squeezy_criterium)
                        
                        
                        #-----------------------------------------------
                        # LA ESTADISTICA SE APLICA EN EL REPORTE!!
                        #-----------------------------------------------
                        values = statistics(values, statistic=statistic)
                        
                        
                        report += '{}_{},{}\n'.format(metric,statistic,values)
                        
                    
        
        if save:
            with open(os.path.join(self.fullpath, 'repetitions_summary.csv'), 'w') as fp:
                fp.write(report)
        
        if show:
            print('\n################################\n')
            print(report)
            print('\n################################\n')
    
    #====================================================
    
    
    
    #====================================================
    def plot_report(self):
        '''
        '''
        
        #----------------------------------------
        # CONSTRUYO TABLA PARA REPORTE FINAL
        #----------------------------------------
        #self.build_report(show=False, save=True)
        
        
        #----------------------------------------
        # GUARDO GRAFICOS REPRESENTATIVOS
        #----------------------------------------
        
        ###############################################
        # TRAIN - PLOT EVOLUTION OF SEVERAL MEASURES
        ###############################################
        MEASURES = ['NUMERO_COEFICIENTES_SELECCIONADOS',
                    'FITNESS',
                    'SHARED_FITNESS',
                    'OBJETIVO_0',
                    'OBJETIVO_1',
                    'OBJETIVO_2',
                    'DISTANCIAS_MEDIAS',
                    'MEDIDA_PARA_ELEGIR_EL_MEJOR_R1',
                    'MEDIDA_PARA_ELEGIR_EL_MEJOR_R2',
                    'CANTIDAD_DE_MUTACIONES',
                    'CANTIDAD_DE_CLUSTERS']
        
        self.plot_summary_train(measures=MEASURES, show=False, save=True)
        
        self.plot_objectives_by_criterium(criterium=['R1', 'R2'])
        
        ###############################################
        # TEST - PLOT EVOLUTION OF SEVERAL MEASURES
        ###############################################
        #MEASURES = ['MEDIDA_PARA_ELEGIR_EL_MEJOR_R1',
                    #'MEDIDA_PARA_ELEGIR_EL_MEJOR_R2']
                    ##'CRITERIO_R1',
                    ##'CRITERIO_R2']
                    ##'NUMERO_COEFICIENTES_SELECCIONADOS',
                    ##'FITNESS',
                    ##'SHARED_FITNESS',
                    ##'OBJETIVO_0',
                    ##'OBJETIVO_1',
                    ##'OBJETIVO_2',
                    ##'DISTANCIAS_MEDIAS',
                    ##'CANTIDAD_DE_MUTACIONES',
                    ##'CANTIDAD_DE_CLUSTERS']
        
        #self.plot_summary_test(measures=MEASURES, show=False, save=True)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




#=============================================
if __name__ == '__main__':
    
    
    import sys
    
    root_path = sys.argv[1]
    
    procesar_replicas(root_path)
