import numpy as np
import os
import sys
from StringIO import StringIO
from _mysql import NULL

expLabel = sys.argv[1]
sizeThreshold = int(sys.argv[2])

clump_file = open(os.path.join(os.path.join(expLabel,'temp_files'),expLabel+'_clumps_final.csv'),'r')
minute_file = open(os.path.join(expLabel,expLabel+'_counts.csv'),'r')

path = expLabel
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

minuteClumps=[]
threshold=sizeThreshold
minutes=np.genfromtxt(minute_file,delimiter=",")

clump_line=clump_file.readline() 
emptyFile=False
clump= np.genfromtxt(StringIO(clump_line),delimiter=',')
for line in minutes:
    minute = line[0]
    minuteClumps.append([minute,[]])
    while clump_line != "" and clump_line != "\n" and len(clump)>0 and clump[0] < minute and not emptyFile:
        clump_line=clump_file.readline()
        if clump_line == NULL or len(clump_line)<3:
            emptyFile=True
        else:
            clump= np.genfromtxt(StringIO(clump_line),delimiter=',')
    while clump_line != "" and clump_line != "\n" and len(clump)>0 and clump[0] == minute and not emptyFile:
        if clump[4]>=threshold:
            minuteClumps[-1][1].append(clump[4])
        clump_line=clump_file.readline()
        if clump_line == NULL or len(clump_line)<3:
            emptyFile=True
        else:
            clump= np.genfromtxt(StringIO(clump_line),delimiter=',')

f = open(os.path.join(expLabel,expLabel+'_clump_counts.csv'),'w')
for i in range(len(minuteClumps)):
    f.write(str(minuteClumps[i][0])+","+str(float(len(minuteClumps[i][1]))/5.0)+"\n")
f.close()
