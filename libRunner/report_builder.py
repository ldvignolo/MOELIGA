import sys
import os
import glob
import numpy as np
import pandas as pd
import yaml
import xlsxwriter # no es necesario importarlo, pero lo hago para que si no esta salte el error al inicio y no despues


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
        
        for i,R in enumerate(RESULTADOS):
            
            if (i < len(parameters)):
                fp.write(R + '\n')
            else:
                fp.write(R[:-1] + '\n')
        
        
    df = pd.read_csv(os.path.join(root_path, 'final_report.csv'),
                     delimiter=',',
                     header=[i for i in range(len(parameters))],
                     index_col=0)
    
    # df.to_excel(os.path.join(root_path, 'final_report.xlsx'))
    
    #---------------------------------------------------------------
    
    idxs = [idx for idx,label in enumerate(df.index) if 'CRITERIO' in label]
    
    sheets = []
    for i in range(len(idxs)):
        
        if (i+1 == len(idxs)):
            sheets.append(pd.DataFrame(df[idxs[i]+1:], columns=df.columns))
        
        else:
            sheets.append(pd.DataFrame(df[idxs[i]+1:idxs[i+1]], columns=df.columns))
        
    #from pandas.io.formats import excel
    #excel.header_style = None    

    # Create a Pandas Excel writer using XlsxWriter as the engine.    
    writer = pd.ExcelWriter(os.path.join(root_path, 'final_report.xlsx'), engine='xlsxwriter')
    workbook  = writer.book
    
    # formato1 = workbook.add_format({'align': 'left'})
    formato1 = workbook.add_format({'bold': True, 'text_wrap': True, 'align': 'left', 'fg_color': '#D7E4BC', 'border': 1, 'bg_color':'#B1B3B3'})
    #formato1 = workbook.add_format()
    #formato1.set_align('left')
    formato2 = workbook.add_format({'num_format': '#,####0.0000', 'align': 'center'})
        
    for i,sheet in enumerate(sheets):        
       
        label = df.index[idxs[i]]
        
        sheet.to_excel(writer, sheet_name='{}'.format(label), header=True, startcol=0)
        
        worksheet = writer.sheets[label]        
                
        first_col_width=0
        for idx, col in enumerate(df.index):  
            if (len(col)>first_col_width):
                first_col_width = len(col)+4            
                
        worksheet.set_column('A:A', first_col_width, formato1)          

        for idx, col in enumerate(df):  
            col_width = len(col[-1])+2
            j=idx+1
            worksheet.set_column(j, j, col_width, formato2)      
            
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    
    
    
    
    
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@






#=============================================
if __name__ == '__main__':
    
    
    import sys
    
    root_path = sys.argv[1]
    
    construir_reporte(root_path)
