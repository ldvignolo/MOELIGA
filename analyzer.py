from libRunner.single_experiment import procesar_experimentos
from libRunner.multiple_runs import procesar_replicas
from libRunner.report_builder import construir_reporte

#=====================================================
import argparse

parser = argparse.ArgumentParser(description='Tool to analyze multiple experiments build with "runner.py".')
parser.add_argument('-p', '--root_path', default='out/', help='Path to output results. By default it is "out/".')
parser.add_argument('-s', '--experiment_settings', default='runner_settings.yaml', help='Settings file to configure this tool.')
parser.add_argument('-l', '--library_path', default='libRunner/', help='Specify path to libRunner library.')
args = vars(parser.parse_args())
#=====================================================




# PROCESAR EXPERIMENTOS INDEPENDIENTES
print('\n################################')
print('Procesando experimentos...')
print('################################')
procesar_experimentos(args['root_path'], lib_path=args['library_path'])
print('Done!!\n')


# PROCESAR REPLICAS EN CONJUNTO Y SUMARIZAR RESULTADOS
print('\n################################')
print('Analizando replicas...')
print('################################')
procesar_replicas(args['root_path'], lib_path=args['library_path'])
print('Done!!\n')


# ANALIZAR REPLICAS Y CONSTRUIR REPORTE
print('\n################################')
print('Construyendo reporte...')
print('################################')
construir_reporte(args['root_path'], args['experiment_settings'])
print('Done!!\n')

print('--------------------------------\n\n')

#=============================================================================
