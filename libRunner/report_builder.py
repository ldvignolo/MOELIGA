import sys
import os
import glob
import numpy as np
import pandas as pd
import yaml


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def procesar_df(parameters, df_name, RESULTADOS):
    '''
    
    '''
    
    update_results = False
    if RESULTADOS:
        update_results = True
        
    
    
    with open(df_name, 'r') as fp:
        lines = fp.read()
        lines = lines.strip().split('\n')
    
    
    data = []
    analyzed_parameters = []
    values = []
    
    # SPLITTING DATA
    flag = False
    for line in lines:
        
        if(line[0] != ','):
            flag = True  # EMPIEZA LA SECCION DE DATOS
        
        if flag:
            data.append(line.strip())
        
        else:
            k,v = line.strip().strip(',').split('=')
            analyzed_parameters.append(k)
            values.append(v)
    
    
    
    
    # CABECERA
    for idx,parameter in enumerate(parameters):
        
        if (len(RESULTADOS) < idx+1):
            RESULTADOS.append('')
            
        if (parameter not in analyzed_parameters):
            analyzed_parameters.insert(idx,'')
            values.insert(idx,'')
            RESULTADOS[idx] += ',{}=None'.format(parameter)
        
        else:
            RESULTADOS[idx] += ',{}={}'.format(analyzed_parameters[idx],values[idx])
    
    
    # DATOS
    for i,d in enumerate(data):
        
        if (len(RESULTADOS) < idx+i+2):
            RESULTADOS.append('')
        
        if update_results:
            _,d = d.split(',')
            RESULTADOS[i+idx+1] += d + ','
        else:
            RESULTADOS[i+idx+1] += d + ','
    
    return RESULTADOS
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@










#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def construir_reporte(root_path, runner_settings):
    '''
    Esta funcion recorre todas las carpetas desde el root del experimento
    completo y procesa cada corrida generando las gráficas y el summary de
    cada experimento ("experiment_summart.json").
    
    '''
    
    #-------------------------------------------
    with open(runner_settings, 'r') as fp:
        if (sys.version_info.major < 3) or (sys.version_info.minor < 6):
            settings = yaml.load(fp)  # 2.7 and < 3.6
        
        else:
            setings = yaml.load(fp, Loader=yaml.FullLoader)
    
    parameters = list(setings['PARAMETERS'].keys())
    #-------------------------------------------
    
    
    paths_to_subfolders = [x[0] for x in os.walk('{}'.format(root_path))]
    
    #-----------------------------------
    # EXTRAIGO PATHS DE LAS REPLICAS
    #-----------------------------------
    RESULTADOS = []
    
    for path in paths_to_subfolders:
        
        filename = glob.glob(os.path.join(path,'repetitions_summary.csv'))  # Devuelve una lista
        
        if filename:
            
            print('\nProcesando {}...'.format(path))
            
            RESULTADOS = procesar_df(parameters, filename[0], RESULTADOS)
        
    
    with open(os.path.join(root_path, 'final_report.csv'), 'w') as fp:
        
        for R in RESULTADOS:
            fp.write(R[:-1] + '\n')
        
        
    df = pd.read_csv(os.path.join(root_path, 'final_report.csv'),
                     delimiter=',',
                     header=[i for i in range(len(parameters))],
                     index_col=0)
    
    df.to_excel(os.path.join(root_path, 'final_report.xlsx'))
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@






#=============================================
if __name__ == '__main__':
    
    
    import sys
    
    root_path = sys.argv[1]
    
    construir_reporte(root_path)
