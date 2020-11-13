import os
import sys
import yaml
import pandas as pd
import itertools

from tqdm import tqdm

#root_path = '../GCM'
#runner_settings = '../runner_settings.yaml'


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def construir_reporte(root_path, runner_settings):
    
    print('\n{}\n\n'.format('='*30))
    
    EXPERIMENTOS = []
    FOLDERS = []

    #---------------------------------------
    # LOAD EXPERIMENT SETTINGS FILE
    #---------------------------------
    with open(runner_settings, 'r') as f:
        SETTINGS_RUNNER = yaml.load(f, Loader=yaml.FullLoader)

    #---------------------------------------    

    names = list(SETTINGS_RUNNER['PARAMETERS'].keys())

    L = [SETTINGS_RUNNER['PARAMETERS'][name] for name in names]

    experiments = list(itertools.product(*L))


    #======================================
    # BUILDING EXPERIMENT SEQUENCE
    #======================================
    print('\nConstruyendo listado de experimentos realizados...')
    for experiment in tqdm(experiments):
        
        folder = root_path + '/'
        #-------------------------------------------

        if not os.path.exists(folder):
            os.makedirs(folder)
            
        parameters = dict()

        # FILTRO LO QUE NO SE MODIFICA POR FALSE
        USAR = dict()

        for name,value in zip(names,experiment):
            
            # NO CARGUE PREVIAMENTE EL VALOR
            if name not in USAR:
                
                USAR[name] = True
                
                if (name in SETTINGS_RUNNER['DEPENDENCIAS']) and (value == False):
                    
                    for key in SETTINGS_RUNNER['DEPENDENCIAS'][name]:
                        
                        USAR[key] = False
            
        #---------------------------------------


        for k,v in zip(names, experiment):
            
            if USAR[k]:
                
                folder += '{}__{}/'.format(k,v)
                parameters[k] = v
                
                folder = folder.replace('True', 'true')
                folder = folder.replace('False', 'false')


        #########################################################
        # EVITO REPETIR EXPERIMENTOS POR CARPETAS REPETIDAS!!
        #########################################################
        if folder not in FOLDERS:
            FOLDERS.append(folder)
            filename = os.path.join(folder, 'repetitions_summary.csv')
            if os.path.isfile(filename):
                EXPERIMENTOS.append(filename)
                
    EXPERIMENTOS = list(set(EXPERIMENTOS))
    EXPERIMENTOS.sort()


    RESULTADOS = []
    
    print('\nExtrayendo información de los experimentos...')
    for path_to_file in tqdm(EXPERIMENTOS):
        
        with open(path_to_file, 'r') as fp:
            lines = fp.read()
            lines = lines.strip().split('\n')
        
        
        analyzed_parameters = dict()
        data = []
        
        #--------------------------
        # SEPARO DATOS EN PARTES
        #--------------------------
        flag = False
        
        for line in lines:
            
            if (line[0] == ','):  # CABECERA
                k,v = line.strip().strip(',').split('=')
                analyzed_parameters[k] = v
            
            else:
                k,v = line.strip().split(',')
                data.append([k,v])
                
        #--------------------------
        
        
        #--------------------------
        # CABECERA
        #--------------------------
        CABECERA = []
        for key in SETTINGS_RUNNER['PARAMETERS'].keys():  # claves en "SETTINGS_RUNNER['PARAMETERS']"
            
            if key in analyzed_parameters.keys():
                CABECERA.append(',{}={}'.format(key, str(analyzed_parameters[key])))
            
            else:
                CABECERA.append(',{}=None'.format(key, None))
        #--------------------------
        
        
        #--------------------------
        # DATOS
        #--------------------------
        #CLAVES = []
        #VALORES = []
        #for (k,v) in data:
            #CLAVES.append('{}'.format(k))
            #VALORES.append('{}'.format(v))
        #--------------------------
        
        if RESULTADOS:  # ACTUALIZO
            for i,label in enumerate(CABECERA):
                RESULTADOS[i] += '{}'.format(label)
            
            if not data:
                print(path_to_file)
            
            for j, (k,v) in enumerate(data):
                RESULTADOS[j+i+1] += ',{}'.format(v)
            
            
        
        else: # INICIALIZO
            RESULTADOS.extend(CABECERA)
            for (k,v) in data:
                RESULTADOS.append('{},{}'.format(k,v))



    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    with open(os.path.join(root_path, 'final_report.csv'), 'w') as fp:
        
        for R in RESULTADOS:
            fp.write(R + '\n')


    df = pd.read_csv(os.path.join(root_path, 'final_report.csv'),
                    delimiter=',',
                    header=[i for i in range(len(parameters)+1)],
                    index_col=0)

    #df.to_excel(os.path.join(root_path, 'final_report.xlsx'))
    #---------------------------------------------------------------

    idxs = [idx for idx,label in enumerate(df.index) if (not isinstance(label, float)) and ('CRITERIO' in label)]

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
    
    
    print('\nEscribiendo reporte...')
    for i,sheet in enumerate(sheets):
        
        print('\nHoja {}...'.format(i+1))
        
        label = df.index[idxs[i]]
        
        sheet.to_excel(writer, sheet_name='{}'.format(label), header=True, startcol=0)
        
        worksheet = writer.sheets[label]        
                
        first_col_width=0
        for idx, col in enumerate(df.index):
            if (not isinstance(col, float)) and (len(col)>first_col_width):
                first_col_width = len(col)+4            
                
        worksheet.set_column('A:A', first_col_width, formato1)          

        for idx, col in enumerate(df):
            col_width = len(col[-1])+2
            j=idx+1
            worksheet.set_column(j, j, col_width, formato2)      
            
        
        print('Done!!!')
        
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    print('\n{}\n\n'.format('='*30))
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@






#=============================================
if __name__ == '__main__':
    
    
    import sys
    
    root_path = sys.argv[1]
    runner_settings = sys.argv[2]
    
    construir_reporte(root_path, runner_settings)
