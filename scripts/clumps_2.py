import numpy as np
import os
import sys
from StringIO import StringIO

expLabel = sys.argv[1]
maxThreshold= int(sys.argv[2])
meanThreshold=int(sys.argv[3])

clump_file = open(os.path.join(os.path.join(expLabel,'temp_files'),expLabel+'_clump_corrected.csv'),'r')
minute_file = open(os.path.join(expLabel,expLabel+'_counts.csv'),'r')

path = os.path.join(expLabel,'temp_files')
datefolder=path
filename = 'clumps_final'
filetype = '.csv'
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

path = os.path.join(path,expLabel+'_'+filename)
f = open(path+filetype,'w')

minutes= np.genfromtxt(minute_file,delimiter=',')
clump_line=clump_file.readline()
clump= np.genfromtxt(StringIO(clump_line),delimiter=',')
for line in minutes:
    minute = line[0]
    while len(clump_line)>2 and (int(clump[0]) < int(minute)):
        clump_line=clump_file.readline()
        if len(clump_line)>2:
            clump= np.genfromtxt(StringIO(clump_line),delimiter=',')
    while len(clump_line)>2 and (int(clump[0]) == int(minute)):
        keepClump=True
        
        if clump[7]<maxThreshold or clump[8]<meanThreshold:
            keepClump=False
        
        if keepClump:
            f = open(path+filetype,'a')
            f.write(str(clump[0]))
            for i in range(1,len(clump)):
                f.write(','+str(clump[i]))
            f.write('\n')
            f.close()
        clump_line=clump_file.readline()
        if len(clump_line)>2:
            clump= np.genfromtxt(StringIO(clump_line),delimiter=',')