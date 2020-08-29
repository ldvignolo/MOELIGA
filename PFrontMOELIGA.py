

def getPFronts(filename):

    # Initialize a dictionary
    dict = {}

    dict[1] = '::> Generacion:' 
    dict[2] = '::> Individuo:'  
    dict[3] = '::> Rank:'       
    dict[4] = '::> Objetivo 0:' 
    dict[5] = '::> Objetivo 1:' 
    dict[6] = '::> Objetivo 2:'

    textfile = open(filename,'r')


    DATA = []
    igen = 0
    rank = 0
    prevind = -1
    indiv = -1
    AUX = []
    obj1=0
    obj2=0
    obj3=0
    flag = False
    for lines in textfile.readlines():
        for eachkey in dict.values():
            if eachkey in lines:

                aux1, aux2 = lines.rstrip()[3:].split(':')
                    
                if eachkey == dict[1]:
                    gen = int(aux2)

                if (gen>igen):
                    DATA.append(AUX)             # agrego todos los individuos del frente en la generacion gen
                    igen+=1
                    AUX = []
                    prevind = -1

                if eachkey == dict[2]:                    
                    indiv = int(aux2)
                    flag = True
                if eachkey == dict[3]:    
                    rank = int(aux2)
                if eachkey == dict[4]:    
                    obj1 = float(aux2)
                if eachkey == dict[5]:    
                    obj2 = float(aux2)
                if eachkey == dict[6]:    
                    obj3 = float(aux2)                                
                    
                    
        if (flag) and (len(lines.split(':'))<=1):    # si encuentro linea vacia es que ya termine de leer el Individuo                    
            if ((rank == 1) and (prevind!=indiv)):                   # solo los individuos del 1er frente
                AUX.append([obj1,obj2,obj3])
                flag = False
                rank = 0
                prevind=indiv
                
            else:
                continue
            
    DATA.append(AUX)             # agrego los de la ultima gen        

    # Close text.txt file
    textfile.close() 
    
   
    return DATA

