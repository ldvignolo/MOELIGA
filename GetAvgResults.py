import numpy as np
import os, sys
from glob import glob
from statistics import mode


if (len(sys.argv)>1):
    ResPath = sys.argv[1]+'/'
else:
    quit()

#ResPath='/home/leandro/Escritorio/nuevo/FS/repo/MOELIGA_dev/resultados_para_borrar_leuk_SVM/'

file_list = [y for x in os.walk(ResPath) for y in glob(os.path.join(x[0], '*.test'))]
nf = len(file_list)

vUARc0=[]
vUARc1=[]
vNFeat=[]

for i,filename in enumerate(file_list):
    ruta, archivo = os.path.split(filename) 
    #print(i+1, ' ', archivo)
    flag = False
    classifier=0
    UAR = [0.0,0.0]
    NFeat = 0
    datafile = open(filename)    
    for line in datafile:
        if 'ELM' in line:
            classifier = 1
            flag = True
        if 'SVM' in line:
            classifier = 0            
        if '> UAR:' in line:
            # print(line.rstrip().split()[1],'',float(line.rstrip().split()[2]))
            UAR[classifier]=float(line.rstrip().split()[2])
        if '> Features:' in line:            
            # print(line.rstrip().split()[1],'',int(line.rstrip().split()[2]))
            NFeat = int(line.rstrip().split()[2])        
    datafile.close()
    vUARc0.append(UAR[0])
    vUARc1.append(UAR[1])
    vNFeat.append(NFeat)
    
    
#print(np.mean(vUARc0))
#print(np.std(vUARc0))
#print(np.mean(vUARc1))
#print(np.std(vUARc1))    
#print(np.mean(vNFeat))
#print(np.std(vNFeat))    
    
outfile = open(ResPath+'results_statistics.txt',"w")
outfile.write("Statistics from "+str(len(vUARc0))+" experiments.\n") 
outfile.write("SVM classifier: UAR Mean: %.2f" % np.mean(vUARc0) + ", Median: %.3f" % np.median(vUARc0)+ ", Mode: %.3f" % mode(vUARc0)+ ", STD: %.3f" % np.std(vUARc0)+"\n") 
if (flag):                                                                                                                                                             
    outfile.write("ELM classifier: UAR Mean: %.2f" % np.mean(vUARc1) + ", Median: %.3f" % np.median(vUARc1)+ ", Mode: %.3f" % mode(vUARc1)+ ", STD: %.3f" % np.std(vUARc1)+"\n") 
outfile.write("No. Features: Mean:       %.1f" % np.mean(vNFeat) + ", Median: %.1f" % np.median(vNFeat)+ ", Mode: %.2f" % mode(vNFeat)+ ", STD: %.2f" % np.std(vNFeat)+"\n") 
outfile.close() 

  
