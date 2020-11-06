from libRunner.single_experiment import procesar_experimentos
from libRunner.multiple_runs import procesar_replicas
from libRunner.report_builder import construir_reporte

#=====================================================
import argparse

parser = argparse.ArgumentParser(description='Tool to analyze multiple experiments build with "runner.py".')
parser.add_argument('-p', '--experiment_path', default='out/', help='Path to output results. By default it is "out/".')
parser.add_argument('-s', '--experiment_settings', default='runner_settings.yaml', help='Settings file to configure this tool.')
parser.add_argument('-l', '--library_path', default='libRunner/', help='Specify path to libRunner library.')
parser.add_argument('-n', '--notification', action='store_true', help='Telegram notification.')
parser.add_argument('-j', '--n_jobs', default=1, help='Number of cores for parallelizing analysis.')
args = vars(parser.parse_args())
#=====================================================



# PROCESAR EXPERIMENTOS INDEPENDIENTES
print('\n################################')
print('Procesando experimentos...')
print('################################')

procesar_experimentos(args['experiment_path'],
                      lib_path=args['library_path'],
                      n_jobs=args['n_jobs'])

print('Done!!\n')


# PROCESAR REPLICAS EN CONJUNTO Y SUMARIZAR RESULTADOS
print('\n################################')
print('Analizando replicas...')
print('################################')

procesar_replicas(args['experiment_path'],
                  lib_path=args['library_path'])

print('Done!!\n')


# ANALIZAR REPLICAS Y CONSTRUIR REPORTE
print('\n################################')
print('Construyendo reporte...')
print('################################')

construir_reporte(args['experiment_path'],
                  args['experiment_settings'])

print('Done!!\n')

print('--------------------------------\n\n')

#try:
    #import openpyxl
#except:
    #print('El módulo "{}" no está disponible.\nInstale el módulo ejecutando pip install {}'.format('openpyxl','openpyxl'))


#=============================================================================

#--------------------------------------------
# Notifico
#--------------------------------------------

if (args['notification']):
    import notification
    notification.notify("La instancia de analyzer " + args['experiment_path'] + " ha finalizado.")

