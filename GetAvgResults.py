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
vTTime=[]

for i,filename in enumerate(file_list):
    # ruta, archivo = os.path.split(filename) 
    #print(i+1, ' ', archivo)
    flag = False
    classifier=0
    UAR = [0.0,0.0]
    NFeat = 0
    TotalTime = 0
    datafile = open(filename)    
    for line in datafile:
        if 'ELM' in line:
            classifier = 1
            flag = True
        if 'SVM' in line:
            classifier = 0            
        if '> UAR:' in line:
            UAR[classifier]=float(line.rstrip().split()[2])
        if '> Features:' in line:            
            NFeat = int(line.rstrip().split()[2])        
    datafile.close()
    vUARc0.append(UAR[0])
    vUARc1.append(UAR[1])
    vNFeat.append(NFeat)
    
    txtfilename = os.path.splitext(filename)[0]+'.txt'
    # print(txtfilename)
    datafile = open(txtfilename)
    for line in datafile:
        if 'TOTAL Time elapsed:' in line:
            TotalTime = float(line.rstrip().split()[3])
    vTTime.append(TotalTime)    
    datafile.close()






try:
    modeUARc0 = mode(vUARc0)
except:
    modeUARc0 = -1
try:
    modeUARc1 = mode(vUARc1)
except:
    modeUARc1 = -1
try:
    modeNFeat = mode(vNFeat)
except:
    modeNFeat = -1
    
    
outfile = open(ResPath+'results_statistics.txt',"w")
outfile.write("Statistics from "+str(len(vUARc0))+" experiments.\n") 
outfile.write("SVM classifier: UAR Mean: %.2f" % np.mean(vUARc0) + ", Median: %.3f" % np.median(vUARc0)+ ", Mode: %.3f" % modeUARc0+ ", STD: %.3f" % np.std(vUARc0)+"\n") 
if (flag):                                                                                                                                                             
    outfile.write("ELM classifier: UAR Mean: %.2f" % np.mean(vUARc1) + ", Median: %.3f" % np.median(vUARc1)+ ", Mode: %.3f" % modeUARc1+ ", STD: %.3f" % np.std(vUARc1)+"\n") 
outfile.write("No. Features: Mean:       %.1f" % np.mean(vNFeat) + ", Median: %.1f" % np.median(vNFeat)+ ", Mode: %.2f" % modeNFeat+ ", STD: %.2f" % np.std(vNFeat)+"\n")
outfile.write("Elapsed Time: Mean:       %.1f" % np.mean(vTTime) + ", Median: %.1f" % np.median(vTTime) + ", STD: %.2f" % np.std(vTTime)+"\n") 
outfile.close() 




  
