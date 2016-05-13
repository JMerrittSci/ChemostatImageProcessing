import os
import sys
import numpy as np

expLabel = sys.argv[1]

path = expLabel
datefolder=path
filename = 'counts'
filetype = '.csv'
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

path = os.path.join(expLabel,expLabel+'_' + filename)
f = open(path+filetype,'w')

data_file = open(os.path.join(os.path.join(expLabel,'temp_files'),expLabel+'_minute_variance_flags.csv'),'r')

data=np.genfromtxt(data_file, delimiter=",")
minutes=data[:,0]
counts=data[:,1]
stdevs=data[:,2]
flags=data[:,3]

endOfFile=False
averageIndices=[]
for i in range(len(minutes)):
    if flags[i]<4:
        averageIndices.append(i)
        if len(averageIndices)>4:
            break

f.write(str(minutes[averageIndices[0]])+","+str(counts[averageIndices[0]])+","+str(stdevs[averageIndices[0]])+"\n")  
f.write(str(minutes[averageIndices[1]])+","+str(counts[averageIndices[1]])+","+str(stdevs[averageIndices[1]])+"\n")  
f.write(str(minutes[averageIndices[2]])+","+str(counts[averageIndices[2]])+","+str(stdevs[averageIndices[2]])+"\n")  
f.write(str(minutes[averageIndices[3]])+","+str(counts[averageIndices[3]])+","+str(stdevs[averageIndices[3]])+"\n")  
f.write(str(minutes[averageIndices[4]])+","+str(counts[averageIndices[4]])+","+str(stdevs[averageIndices[4]])+"\n")  

numValidMinutes=5

for i in range(averageIndices[4]+1,len(minutes)):
    mean = np.mean(counts[averageIndices])
    stdev= np.max([np.std(averageIndices),np.sqrt(mean)])
    if (np.fabs(counts[i]-mean)<3*stdev and flags[i]<5) or (counts[i]>0 and (i-averageIndices[0])>9):
        f.write(str(minutes[i])+","+str(counts[i])+","+str(stdevs[i])+"\n") 
        averageIndices.pop(0)
        averageIndices.append(i)
        numValidMinutes+=1
    elif np.fabs(counts[i]-mean)<4*stdev and flags[i]<4:
        f.write(str(minutes[i])+","+str(counts[i])+","+str(stdevs[i])+"\n") 
        averageIndices.pop(0)
        averageIndices.append(i)
        numValidMinutes+=1
f.close()