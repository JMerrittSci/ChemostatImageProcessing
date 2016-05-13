import os
import numpy as np
import sys
from StringIO import StringIO

expLabel = sys.argv[1]
firstMinute = int(sys.argv[2])

path = os.path.join(expLabel,'temp_files')
datefolder=path
filename = 'minute_variance_flags'
filetype = '.csv'
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

path = os.path.join(path,expLabel + '_' + filename)
f = open(path+filetype,'w')
data_file = open(os.path.join(os.path.join(expLabel,'temp_files'),expLabel+'_frame_counts_flags.csv'),'r')

minute=firstMinute
counts=[]
flags=[]
endOfFile=False
while not endOfFile:
    for i in range(5):
        try:
            line=data_file.readline()
            if line is None or len(line)<3:
                endOfFile=True
                break
            line=np.genfromtxt(StringIO(line), delimiter=",")
            if len(line)==0:
                endOfFile=True
                break
            counts.append(line[2])
            flags.append(line[3])
        except:
            endOfFile=True
            break
    removeFrames=[]
    for i in reversed(range(len(flags))):
        if flags[i]>3:
            removeFrames.append(i)
    for i in removeFrames:
        flags.pop(i)
        counts.pop(i)
    if len(flags)>2:
        flagAv=np.mean(flags)
    else:
        flagAv=4
    if len(flags)>2 and flagAv<2.5:
        cMean=np.mean(counts)
        cStdev=np.std(counts)
        fraction=cStdev/np.sqrt(cMean)
        if fraction>1.9:
            f.write(str(minute)+","+str(cMean)+","+str(cStdev)+",5\n") 
        elif fraction>1.8:
            f.write(str(minute)+","+str(cMean)+","+str(cStdev)+",4\n") 
        elif fraction>1.7:
            f.write(str(minute)+","+str(cMean)+","+str(cStdev)+",3\n") 
        elif fraction>1.6:
            f.write(str(minute)+","+str(cMean)+","+str(cStdev)+",2\n") 
        elif fraction>1.5:
            f.write(str(minute)+","+str(cMean)+","+str(cStdev)+",1\n") 
        else:
            f.write(str(minute)+","+str(cMean)+","+str(cStdev)+",0\n") 
    else:
        f.write(str(minute)+",0,0,5\n")
    
    minute+=1
    counts=[]
    flags=[]

f.close()