import os
import glob
import numpy as np
import json
import yaml
import pandas as pd
import io

#import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from celluloid import Camera

from scipy.stats import median_absolute_deviation
from scipy.stats import mode

import inspect






#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def procesar_experimentos(root_path, lib_path):
    '''
    Esta funcion recorre todas las carpetas desde el root del experimento
    completo y procesa cada corrida generando las gráficas y el summary de
    cada experimento ("experiment_summart.json").
    
    '''
    
    paths_to_subfolders = [x[0] for x in os.walk('{}'.format(root_path))]
    
    
    for path in paths_to_subfolders:
        
        filename = glob.glob(os.path.join(path,'*.train'))  # Devuelve una lista
        
        if filename:
            
            print('\nProcesando {}...'.format(path))
            
            se = SINGLE_EXPERIMENT(filename[0], lib_path=lib_path)
            
            se.build_summary()
            
            # GUARDAR OBJETO CON PICKLE O DILL??
            
            del se
            
            print('Done!!\n')
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@













#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class SINGLE_EXPERIMENT(object):
    '''
    Extrae la información de un experimento y la ordena para acceder de forma sencilla.
    '''
    
    def __init__(self, filename, lib_path, verbose=False):
        '''
        '''
        
        self.lib_path = lib_path
        
        full_path = os.path.abspath(filename)
        
        self.path, self.name = os.path.split(full_path)
        
        if ('.train' in self.name):
            self.name = self.name.replace('.train', '')
        
        if ('.test' in self.name):
            self.name = self.name.replace('.test', '')
        
        
        filename_train = self.name + '.train'
        filename_test = self.name + '.test'
        
        
        #===================
        # LOAD TRAIN DATA
        #===================
        
        #---------------------------------------------------------------
        with open(os.path.join(self.path,filename_train),'r') as fp:
            train_report = yaml.load(fp, Loader=yaml.FullLoader)
        
        self.generations = []
        self.general = dict()
        self.pareto = dict()
        
        self.test = dict()
        
        #----------------------------------
        for GENERATION in train_report:
            
            #--------------
            # GENERATIONS
            #--------------
            self.generations.append(GENERATION['GENERATION'])
            
            
            #-----------
            # GENERAL
            #-----------
            CURRENT_GENERATION = dict()
            
            for k,v in GENERATION['GENERAL'].items():
                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # CORRECCION DE CLAVES
                if (k == 'TOTAL_ELAPSED_TIME'):
                    k = 'ELAPSED_TIME'
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                # AGREGO LA CLAVE SI NO ESTÁ
                if (k not in CURRENT_GENERATION.keys()):
                    CURRENT_GENERATION[k] = v
            
            
            # ACTUALIZO HISTORIAL GENERAL
            for k,v in CURRENT_GENERATION.items():
                
                if (k not in self.general.keys()):
                    self.general[k] = []
                    
                self.general[k].append(v)
            
            
            
            #-----------
            # PARETO
            #-----------
            pareto = dict()
            
            #------------------------------------------------------
            for individual in GENERATION['FRENTE_DE_PARETO']:
                
                for k,v in individual.items():
                    
                    #----------------------------
                    if (k != 'OBJETIVOS'):
                        if (k not in pareto.keys()):
                                pareto[k] = []
                        
                        pareto[k].append(v)  # ACTUALIZO FRENTE ACTUAL
                    
                    #----------------------------
                    else:
                        
                        for i in range(len(v)):
                            
                            k = 'OBJETIVO_{}'.format(i)
                            
                            if (k not in pareto.keys()):
                                pareto[k] = []
                            
                            pareto[k].append(v[i])  # ACTUALIZO FRENTE ACTUAL
                    #----------------------------
                
            
            #------------------------------------------------------
            # ACTUALIZO FRENTE HISTORICO
            #-----------------------------
            for k,v in pareto.items():
                
                if (k not in self.pareto.keys()):
                    self.pareto[k] = []
                
                self.pareto[k].append(v)
            #------------------------------------------------------
            
        
        
        #================
        # TEST DATA
        #================
        
        with open(os.path.join(self.path,filename_test),'r') as fp:
            self.test = json.load(fp)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
    
    
    #====================================================
    def _statistics(self, data, statistic=None, axis=0):
        '''
        STATISTICS: [mean, median, std, mad, max, min, mode] --> axis=0
        '''
            
        
        #-------------------------------------
        # STATISTIC
        #------------
        if (statistic == 'mean'):
            data = np.mean(data, axis=axis)
        
        elif (statistic == 'median'):
            data = np.median(data, axis=axis)
        
        elif (statistic == 'std'):
            data = np.std(data, axis=axis)
        
        elif (statistic == 'mad'):
            values = median_absolute_deviation(data, axis=axis)
            data = values[0].flatten()
        
        elif (statistic == 'max'):
            data = np.max(data, axis=axis)
        
        elif (statistic == 'min'):
            data = np.min(data, axis=axis)
        
        elif (statistic == 'mode'):
            data = mode(data, axis=axis)
        
        if isinstance(data, np.ndarray):
            data = data.tolist()
        
        return data
    #====================================================
    
    
    #====================================================
    def apply_statistic(self, data, statistic=None, squeezy_criterium=None):
        '''
        SQUEEZING: [mean, median, max, min, mode] --> axis=1
        
        STATISTICS: [mean, median, std, mad, max, min, mode] --> axis=0
        '''
        
        if (squeezy_criterium != None):
            data = self._statistics(data, statistic=squeezy_criterium, axis=1)
        
        if (statistic != None):
            data = self._statistics(data, statistic=statistic, axis=0)
        
        if isinstance(data, np.ndarray):
            data = data.tolist()
        
        return data        
    #====================================================
    
    
    #====================================================
    def from_confusion_matrix_get_measures(self, M, as_list=True):
        '''
        '''
        
        
        if not isinstance(M, np.ndarray):
            M = np.array(M).astype(float)
        
        measures = {key:np.zeros(M.shape[0]) for key in ['TP',
                                                         'TN',
                                                         'FP',
                                                         'FN',
                                                         'PRECISION',
                                                         'RECALL',
                                                         'SPECIFICITY',
                                                         'ACCURACY',
                                                         'UAR',
                                                         'F1',
                                                         'F1_MACRO',
                                                         'F1_WEIGHTED',
                                                         'KAPPA_SCORE']}
        
        
        
        #----------------------------------
        # KAPPA_SCORE
        #-------------
        #https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c
        #----------------------------------
        N = M.sum()
        a = M.sum(axis=0) / N
        b = M.sum(axis=1) / N
        measures['KAPPA_SCORE'] = np.sum(a * b)
        
        
        
        
        measures['TP'] = M.diagonal()
        measures['TN'] = measures['TP'].sum() - measures['TP']
        
        #----------------------------
        # ESTRUCTURA:
        # rows --> Real
        # cols --> Predicho
        measures['FP'] = M.sum(axis=0) - measures['TP']
        measures['FN'] = M.sum(axis=1) - measures['TP']
        #----------------------------
        
        for j in range(measures['TP'].size):
            
            TP = measures['TP'][j]
            TN = measures['TN'][j]
            FP = measures['FP'][j]
            FN = measures['FN'][j]
            
            
            #-----------------
            # PRECISION (PPV)
            #-----------------
            if ((TP + FP) != 0):
                measures['PRECISION'][j] = TP / (TP + FP)
            
            
            #-----------------
            # RECALL (TPR)
            #-----------------
            if ((TP + FN) != 0):
                measures['RECALL'][j] = TP / (TP + FN)
            
            
            #-------------------
            # SPECIFICITY (TNR)
            #-------------------
            if ((TN + FP) != 0):
                measures['SPECIFICITY'][j] = TN / (TN + FP)
            
            #-------------------
            # ACCURACY (ACC)
            #-------------------
            if ((TP + TN + FP + FN) != 0):
                measures['ACCURACY'][j] = (TP + TN) / (TP + TN + FP + FN)
            
            
            #-------------------
            # UAR
            #-------------------
            measures['UAR'][j] = (measures['RECALL'][j] + measures['SPECIFICITY'][j]) / 2.0
            
            
            #-----------------
            # F1
            #-----------------
            if ((TP + FP + FN) != 0):
                measures['F1'][j] = (2.0 * TP) / (2.0 * TP + FP + FN)
            
        
        
        #-------------------
        # UAR
        #-------------------
        measures['UAR'] = np.mean(measures['UAR'])
        
        #-----------------
        # F1_MACRO
        #-----------------
        measures['F1_MACRO'] = np.mean(measures['F1'])
        
        #-----------------
        # F1_WEIGHTED
        #-----------------
        measures['F1_WEIGHTED'] = np.sum(measures['F1'] * M.sum(axis=1)) / N
        
        
        
        
        if as_list:
            for key in measures.keys():
                measures[key] = measures[key].tolist()
        
        return measures
    #====================================================
    
    
    #==========
    # PARETO
    #==========
    
    
    ##====================================================
    #def from_pareto_get_xxx(self, G=None):
        #'''
        #'''
        
        #key = inspect.stack()[0][3]  # Devuelve el nombre de la funcion
        #key = key.replace('from_pareto_get_','').upper()

        #data = self._procesar_pareto(key, G)
        
        #return data
    ##====================================================
    
    
    #====================================================
    def from_pareto_get_best_individual(self, criterium='R1'):
        '''
        criterium: ['R1' | 'R2']
        
        '''
        
        #key = inspect.stack()[0][3]  # Devuelve el nombre de la funcion
        #key = key.replace('from_pareto_get_index_for_best_','').upper()
        #data = self._procesar_pareto(key, -1)
        
        if (criterium == 'R1'):
            measures = self.pareto['MEDIDA_PARA_ELEGIR_EL_MEJOR_R1'][-1][:]
        elif (criterium == 'R2'):
            measures = self.pareto['MEDIDA_PARA_ELEGIR_EL_MEJOR_R2'][-1][:]
        
        else:
            print('Unknown criterium...\n')
        
        idx = int(np.argmax(measures))
        
        return idx
    #====================================================
    
    

    
    #==========
    # TEST
    #==========
    
    
    
    #===============================
    # PLOT GENERAL/ELITE MEASURES
    #===============================
    
    
    #====================================================
    def plot_measure_evolution(self, measure, measure_name, ax=None, show=True, save=False):
        '''
        '''
        
        G = self.generations[:]
        
        if (ax == None):
            fig, ax = plt.subplots(1, 1, figsize=(8,2))
        
        #-----------------
        # DATA
        #-----------------
        if (measure['raw'] == None):
            
            l1 = ax.plot(G, measure['mean'],'-r', linewidth=2)
            l2 = ax.plot(G, measure['median'],':g', linewidth=2)
            
            ax.fill_between(G,
                            measure['max'],
                            measure['min'],
                            facecolor='yellow', alpha=0.5)
        
        else:
            l1 = ax.plot(G, measure['raw'], '-r', linewidth=2)
        
        ax.set_title(u'Evolution of {}'.format(measure_name), fontsize=7)
        ax.set_xlabel(u'Generations', fontsize=7)
        ax.set_ylabel(u'Measure', fontsize=7)
        
        ax.grid(True)
        ax.tick_params(axis='x', which='major', labelsize=7)
        ax.tick_params(axis='y', which='major', labelsize=7)
        
        if (measure['raw'] == None):
            ax.legend([r'mean', r'median'], loc='best')
        
        plt.tight_layout()
        #fig.tight_layout(pad=4.0)
        
        
        if save:
            plt.savefig(os.path.join(self.path,'evolution_of_{}.pdf'.format(measure_name)), dpi=600)
            plt.savefig(os.path.join(self.path,'evolution_of_{}.png'.format(measure_name)), dpi=600)
        
        if show:
            plt.show()
        
        if (ax == None):
            plt.close(fig)
    #====================================================
    
    
    #====================================================
    def plot_histogram_evolution_of_features_selected(self, ax=None, show=True, save=False):
        '''
        '''
        
        data = np.array(self.general['NUMERO_DE_VECES_QUE_SE_ELIGE_CADA_FEATURE']).T
        
        
        if (ax == None):
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
        
        #-----------------
        # DATA
        #-----------------
        if isinstance(data, list):
            G = np.arange(0, len(data))
        else:
            G = np.arange(0, data.shape[0])
        
        
        evol_ = ax.imshow(np.log2(data.shape[0]/(data+1)),
                          interpolation='nearest',
                          aspect='auto',
                          cmap=plt.get_cmap('jet'))
        
        ax.set_title(u'Evolution of features selected\n(Higher values indicates selected more times)', fontsize=14)
        ax.set_xlabel(u'Generations', fontsize=12)
        ax.set_ylabel(u'Features', fontsize=12)
        
        ax.grid(True)
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        
        #_ax = plt.gca()
        #PCM = ax.get_children()[2]  # Get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(evol_, ax=ax) 
        
        plt.tight_layout()
        #fig.tight_layout(pad=4.0)
        
        if save:
            plt.savefig(os.path.join(self.path,'evolution_of_features_selected.pdf'), dpi=600)
            plt.savefig(os.path.join(self.path,'evolution_of_features_selected.png'), dpi=600)
        
        if show:
            plt.show()
        
        if (ax == None):
            plt.close(fig)
    #====================================================
    
    
    # NUEVO PLOT
    #De paso, otra gráfica similar que sería interesante hacer también,
    #es la distancia media de los individuos del frente de cada generación.
    #Esa media esta en la info de cada individuo del frente y es un único valor.
    
    
    
    #pareto --> MEAN_DISTANCE: 0.395084
    
    
    
    #====================================================
    def plot_evolution_of_best_UAR(self, criterium='R1', ax=None, show=True, save=False):
        '''
        '''
        
        if (criterium == 'R1'):
            measures = self.pareto['MEDIDA_PARA_ELEGIR_EL_MEJOR_R1']
            
        elif (criterium == 'R2'):
            measures = self.pareto['MEDIDA_PARA_ELEGIR_EL_MEJOR_R2']
        else:
            print('Unknown criterium...\n')
        
        UAR = []
        
        for i,measure in enumerate(measures):
            idx = int(np.argmax(measure))
            UAR.append(self.pareto['OBJETIVO_1'][i][idx])
        
        measure = {'mean':[], 'median':[], 'max':[], 'min':[], 'raw':UAR}
        
        self.plot_measure_evolution(measure, 'UAR [{} criterium]'.format(criterium), ax=ax, show=show, save=save)
        
        
        
    #====================================================
    
    
    #==================================
    # PLOT PARETO FRONT AND MEASURES
    #==================================
    def plot_confusion_matrix(self, criteria=['R1','R2'], show=True, save=False):
        '''
        '''
        
        NR = len(self.test[0]['CLASIFICADORES'])
        NC = len(criteria)
        
        fig, ax = plt.subplots(NR, NC, figsize=(5*NC,2*NR))
        
        for idx_c,criterium in enumerate(criteria):
            
            idx = self.from_pareto_get_best_individual(criterium=criterium)
            
            CLASIFICADORES = self.test[idx]['CLASIFICADORES']
            
            for idx_r,classifier in enumerate(CLASIFICADORES.keys()):
                
                CM = self.test[idx]['CLASIFICADORES'][classifier]['CONFUSION_MATRIX']
                
                sns.heatmap(CM,
                            annot=True,
                            fmt="d",
                            square=True,
                            cbar=False,
                            ax=ax[idx_r,idx_c])#,
                            #cmap='jet')
                
                ax[idx_r,idx_c].set_title('Criterio: {} -- Classifier:{}'.format(criterium, classifier))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.path,'confusion_matrix.pdf'), dpi=600)
            plt.savefig(os.path.join(self.path,'confusion_matrix.png'), dpi=600)
        
        if show:
            plt.show()
        
        plt.close(fig)
        
    
    #====================================================
    def plot_pareto_front_animated(self, show=True, save=False):
        '''
        https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
        '''
        #import numpy as np
        #from matplotlib import pyplot as plt
        from matplotlib.colors import Normalize
        #fig = plt.figure()
        
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        camera = Camera(fig)
        
        _cm = cm.get_cmap('jet')
        
        N = len(self.pareto['OBJETIVO_0'])
        
        for i in range(N):
            
            Obj0 = np.array(self.pareto['OBJETIVO_0'][i])
            Obj1 = np.array(self.pareto['OBJETIVO_1'][i])
            Obj2 = np.array(self.pareto['OBJETIVO_2'][i])
            
            #ax.scatter(Obj0, Obj1, s=0.02**(-(1.+Obj2)), c='C0')
            #ax.scatter(Obj0, Obj1, s=30, c=['r' if i%2 == 0 else 'b' for i in range(Obj0.shape[0])])
            
            ax.scatter(Obj0, Obj1, s=70, c=[_cm((2/(1+(np.exp(-5*item))))-1) for item in Obj2])
            
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])
            ax.set_xlabel('UAR', fontsize=12)
            ax.set_ylabel('Nfeatures', fontsize=12)
            ax.grid(True)
            
            ax.text(x=0.75,
                    y=1.1,
                    s='Generation {}'.format(i),
                    va='center',
                    ha='right',
                    fontsize=12)
        
            camera.snap()
        
        fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1, clip=False), cmap='jet'), ax=ax)
        
        animation = camera.animate()
        
        if show:
            plt.show()
        
        if save:
            animation.save(os.path.join(self.path,'pareto_front_animated.gif'),
                           writer='imagemagick')
    
    
    
    
    
    
    
    #====================================================
    def plot_pareto_front2(self, G=0, show=True, save=False):
        
        #NF = np.round(self.get_pareto_Nfeatures(G) * 10).astype(int)
        #RELIEF = np.round(self.get_pareto_Relief(G) * 10).astype(int)
        #UAR = self.get_pareto_UAR(G)
        
        NF = self.get_pareto_Nfeatures(G)
        RELIEF = self.get_pareto_Relief(G)
        UAR = self.get_pareto_UAR(G)
        
        
        #M = np.zeros((resolution,resolution))
        
        
        # setup the figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(20,6))
        
        ax.scatter(NF, RELIEF, 50, 'C0')#['C{}'.format(n) for n in np.round(self.get_pareto_UAR(G) * 10).astype(int)])
        
        #ax.set_title('Shaded')
        
        ax.set_xlim(-.01,1.01)
        ax.set_ylim(-.01,1.51)
        
        #ax.semilogx()
        #ax.semilogy()
        ax.set_xlabel('Number of features')
        ax.set_ylabel('Relief')
        
        plt.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.path,'pareto_front.pdf'), dpi=600)
            plt.savefig(os.path.join(self.path,'pareto_front.png'), dpi=600)
        
        if show:
            plt.show()
        
        
        plt.close(fig)
    #====================================================
    
    
    
    #====================================================
    def plot_distancia_media_frente_pareto(self, ax=None, show=False, save=False):
        '''
        '''
        
        data = self.from_pareto_get_mean_distance()
        
        if (ax == None):
            fig, ax = plt.subplots(1, 1, figsize=(20,6))
        
        #-----------------
        # DATA
        #-----------------
        G = np.arange(0, len(data))
        
        general_min = []
        general_mean = []
        general_median = []
        general_max = []
        
        for d in data:
            general_min.append(np.min(d))
            general_mean.append(np.mean(d))
            general_median.append(np.median(d))
            general_max.append(np.max(d))
            
            
        l1 = ax.plot(G, general_mean, '-r', linewidth=2)
        l2 = ax.plot(G, general_median, ':g', linewidth=2)
        
        ax.fill_between(G, general_max, general_min, facecolor='yellow', alpha=0.5)
        
        
        ax.set_title(u'Evolution of {}'.format('Evolucion en la distancia media en el frente de Pareto'), fontsize=7)
        ax.set_xlabel(u'Generations', fontsize=7)
        ax.set_ylabel(u'Measure', fontsize=7)
        
        ax.grid(True)
        ax.tick_params(axis='x', which='major', labelsize=7)
        ax.tick_params(axis='y', which='major', labelsize=7)
        
        ax.legend([r'mean', r'median'], loc='best')
        
        plt.tight_layout()
        #fig.tight_layout(pad=4.0)
        
        
        if save:
            plt.savefig(os.path.join(self.path,'Evolucion_en_la_distancia_media_en_el_frente_de_Pareto.pdf'), dpi=600)
            plt.savefig(os.path.join(self.path,'Evolucion_en_la_distancia_media_en_el_frente_de_Pareto.png'), dpi=600)
        
        if show:
            plt.show()
        
        
        if (ax == None):
            plt.close(fig)  # NO OLVIDAR ESTO PARA QUE NO QUEDE CARGADA EN MEMORIA!!!
                            # https://heitorpb.github.io/bla/2020/03/18/close-matplotlib-figures/
    #====================================================
    
    
    
    
    
    
    
    #====================================================
    def plot_summary(self, measures=[], show=True, save=False):
        '''
        '''
        
        N = len(measures) + 2
        
        #--------------------------------------------------------------------
        fig, ax = plt.subplots(N, 1, figsize=(8, 2*N))
        
        for idx, (source,measure_name) in enumerate(measures):
            
            if (source == 'PARETO'):
                
                data = self.pareto[measure_name]
                
                #-------------------------------------------------------------
                if isinstance(data[0], list):
                    
                    measure = {'mean':[], 'median':[], 'max':[], 'min':[], 'raw':None}
                    
                    for d in data:
                        measure['mean'].append(float(np.mean(d)))
                        measure['median'].append(float(np.median(d)))
                        measure['max'].append(float(np.max(d)))
                        measure['min'].append(float(np.min(d)))
                #-------------------------------------------------------------
                else:
                    measure = {'mean':[], 'median':[], 'max':[], 'min':[], 'raw':data}
                #-------------------------------------------------------------
            
                
            #-------------------------------------------------------------
            elif (source == 'GENERAL'):
                
                data = self.general[measure_name]
                
                if isinstance(data[0], list):
                    
                    measure = {'mean':[], 'median':[], 'max':[], 'min':[], 'raw':None}
                    
                    for d in data:
                        measure['mean'].append(float(np.mean(d)))
                        measure['median'].append(float(np.median(d)))
                        measure['max'].append(float(np.max(d)))
                        measure['min'].append(float(np.min(d)))
                #-------------------------------------------------------------
                
                else:
                    measure = {'mean':[], 'median':[], 'max':[], 'min':[], 'raw':data}
                #-------------------------------------------------------------
            
            self.plot_measure_evolution(measure, measure_name=measure_name, ax=ax[idx], show=False, save=False)
            
        #--------------------------------------------------------------------
        
        self.plot_evolution_of_best_UAR(criterium='R1', ax=ax[idx+1], show=False, save=False)
        
        self.plot_evolution_of_best_UAR(criterium='R2', ax=ax[idx+2], show=False, save=False)
        
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
    def build_summary(self):
        '''
        Este método genera el archivo JSON que resume los resultados del experimento.
        El archivo generado puede ser luego leido por una función que procese los
        resultados para varias corridas.
        
        '''
        
        with open(self.lib_path + 'experiment_summary_template.yaml', 'r') as fp:
            TEMPLATE = yaml.load(fp, Loader=yaml.FullLoader)
        
        
        SUMMARY = {
                   'TRAIN': dict(),
                   'TEST': dict()
                  }
        
        
        #===============
        # TRAIN
        #===============
        for key in TEMPLATE['TRAIN'].keys():
            
            squeezy_criterium = TEMPLATE['TRAIN'][key]
            
            if (key == 'ELAPSED_TIME'):
                t = self.general['ELAPSED_TIME'][:]
                t.insert(0,0)
                SUMMARY['TRAIN'][key] = np.diff(t).tolist()
            
            else:
                
                values = self.apply_statistic(self.general[key],
                                              statistic=None,
                                              squeezy_criterium=squeezy_criterium
                                             )
                
                
                if isinstance(values, np.ndarray):
                    SUMMARY['TRAIN'][key] = values.tolist()
                
                elif isinstance(values, list):
                    SUMMARY['TRAIN'][key] = values[:]
                
                elif isinstance(values, np.int64):
                    SUMMARY['TRAIN'][key] = int(values)
                
                elif isinstance(values, np.float64):
                    SUMMARY['TRAIN'][key] = float(values)
                
                elif isinstance(values, int) or isinstance(values, float):
                    SUMMARY['TRAIN'][key] = values
                
                else:
                    print('No se puede guardar el valor para {} [Type: {}]'.format(key, type(values)))
                    
                
        
        
        
        #===============
        # TEST
        #===============
        
        # --> [R1, R2]
        
        for criterium in TEMPLATE['TEST']['CRITERIO']:
            
            idx = 0
            
            if (len(self.test) > 1):
                idx = self.from_pareto_get_best_individual(criterium)
            
            individuo = self.test[idx]
            
            #=======================================================
            
            # GUARDO LAS MEDIDAS DE INTERES
            
            if criterium not in SUMMARY['TEST'].keys():
                SUMMARY['TEST'][criterium] = {'CLASIFICADORES': dict()}
                
            for measure in TEMPLATE['TEST']['MEASURES']:
                
                SUMMARY['TEST'][criterium][measure] = individuo[measure]
                
                
            #-------------------------------------------------------
            
            # GUARDO LOS CLASIFICADORES Y MEDIDAS
            
            for clasificador in individuo['CLASIFICADORES'].keys():
                
                if clasificador not in SUMMARY['TEST'][criterium]['CLASIFICADORES'].keys():
                    SUMMARY['TEST'][criterium]['CLASIFICADORES'][clasificador] = dict()
                
                for key,value in individuo['CLASIFICADORES'][clasificador].items():
                    SUMMARY['TEST'][criterium]['CLASIFICADORES'][clasificador][key] = value
                
                metrics = self.from_confusion_matrix_get_measures(individuo['CLASIFICADORES'][clasificador]['CONFUSION_MATRIX'], as_list=True)
                
                for metric_name,value in metrics.items():
                    
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    
                    SUMMARY['TEST'][criterium]['CLASIFICADORES'][clasificador][metric_name] = value
                
                #-------------------------------------------------------------
        
        #-------------------------------
        # SAVE SUMMARY IN JSON FORMAT
        #-------------------------------
        with open(os.path.join(self.path,'experiment_summary.json'), 'w') as fp:
            json.dump(SUMMARY, fp)
        
        
        #######################################
        # PLOT EVOLUTION OF SEVERAL MEASURES
        #######################################
        MEASURES = [('GENERAL','NUMERO_COEFICIENTES_SELECCIONADOS'),
                    ('GENERAL','FITNESS'),
                    ('GENERAL','SHARED_FITNESS'),
                    ('GENERAL','OBJETIVO_0'),
                    ('GENERAL','OBJETIVO_1'),
                    ('GENERAL','OBJETIVO_2'),
                    ('GENERAL','DISTANCIAS_MEDIAS'),
                    ('GENERAL','MEDIDA_PARA_ELEGIR_EL_MEJOR_R1'),
                    ('GENERAL','MEDIDA_PARA_ELEGIR_EL_MEJOR_R2'),
                    ('GENERAL','CANTIDAD_DE_MUTACIONES'),
                    ('GENERAL','CANTIDAD_DE_CLUSTERS'),
                    ('PARETO','MEAN_DISTANCE')] # --> PARETO]
        
        #--------------------------------------
        # PLOT EVOLUTION OF UAR [R1 y R2]
        #--------------------------------------
        #self.plot_evolution_of_best_UAR(criterium='R2', ax=None, show=False, save=True)
        self.plot_summary(measures=MEASURES, show=False, save=True)
        
        
        #--------------------------------------
        # PLOT EVOLUTION OF FEATURES SELECTED
        #--------------------------------------
        self.plot_histogram_evolution_of_features_selected(show=False, save=True)
        
        #--------------------
        # PLOT PARETO FRONT
        #--------------------
        #self.plot_pareto_front(show=False, save=True)
        
        #------------------------
        # PLOT CONFUSION MATRIX
        #------------------------
        # self.plot_confusion_matrix(criteria=['R1','R2'], show=False, save=True)
        
        
        # ANIMACION DEL FRENTE DE PARETO
        self.plot_pareto_front_animated(show=False, save=True)
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



#=============================================
if __name__ == '__main__':
    
    
    import sys
    
    root_path = sys.argv[1]
    
    procesar_experimentos(root_path)
