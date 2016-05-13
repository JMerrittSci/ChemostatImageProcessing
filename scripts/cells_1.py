import sys
import os
import numpy as np
import math
from StringIO import StringIO

expLabel = sys.argv[1]
firstMinute = int(sys.argv[2])

path = os.path.join(expLabel,'temp_files')
datefolder=path
filename0 = 'frame_counts_flags'
filetype = '.csv'
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

path = os.path.join(path,expLabel+'_' + filename0)
f_flags = open(path+filetype,'w')

data_file = open(os.path.join(os.path.join(expLabel,'datasets'),expLabel+'_dataset.csv'),'r')

endOfFile=False

minute=firstMinute
frame=0
badFrames=0

counts=0
ones=0
x=[]
y=[]
while not endOfFile:
    try:
        line=data_file.readline()
        if line is None or len(line)<3:
            endOfFile=True
            break
        line=np.genfromtxt(StringIO(line), delimiter=",")
        if len(line)==0:
            endOfFile=True
            break
    except:
        endOfFile=True
        break
    if minute == int(line[0]+0.01) and frame == int(line[1]+0.01):
        label=int(line[2]+0.01)
        if label > 0:
            x_pos=int(line[3]+0.01)
            y_pos=int(line[4]+0.01)
            if x_pos > 7 and x_pos < 1280 and y_pos > 7 and y_pos < 956:
                counts+=label
                ones+=1
                for loopVar in range(label):
                    x.append(x_pos)
                    y.append(y_pos)
    else:
        while minute < int(line[0]+0.01) or (minute == int(line[0]+0.01) and frame < int(line[1]+0.01)):
            if len(x)==0:
                f_flags.write(str(minute)+","+str(frame)+",0,0\n")
            else:
                x_mean=np.mean(x)
                y_mean=np.mean(y)
                stdevX=math.sqrt((math.pow(1288,2)-1)/12)/math.sqrt(counts)
                stdevY=math.sqrt((math.pow(964,2)-1)/12)/math.sqrt(counts)
                dispX=math.fabs(x_mean-643.5)
                dispY=math.fabs(y_mean-481.5)
                probX=1-math.erf(dispX/(stdevX*math.sqrt(2)))
                probY=1-math.erf(dispY/(stdevY*math.sqrt(2)))
                prob=probX*probY
                if prob < 0.000001:
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",4\n")
                elif prob <0.00001:
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",3\n")
                elif prob <0.0001:
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",2\n")
                elif prob <0.001:
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",1\n")
                else:
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",0\n")
            frame+=1
            if frame>4:
                frame=0
                minute+=1
            counts=0
            ones=0
            x=[]
            y=[]
        ### minute/frames match
        label=int(line[2]+0.01)
        if label > 0:
            x_pos=int(line[3]+0.01)
            y_pos=int(line[4]+0.01)
            if x_pos > 7 and x_pos < 1280 and y_pos > 7 and y_pos < 956:
                counts+=label
                ones+=1
                for loopVar in range(label):
                    x.append(x_pos)
                    y.append(y_pos)

x_mean=np.mean(x)
y_mean=np.mean(y)
stdevX=math.sqrt((math.pow(1272,2)-1)/12)/math.sqrt(counts)
stdevY=math.sqrt((math.pow(948,2)-1)/12)/math.sqrt(counts)
dispX=math.fabs(x_mean-643.5)
dispY=math.fabs(y_mean-481.5)
probX=1-math.erf(dispX/(stdevX*math.sqrt(2)))
probY=1-math.erf(dispY/(stdevY*math.sqrt(2)))
prob=probX*probY
if prob < 0.000001:
    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",4")
elif prob <0.00001:
    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",3")
elif prob <0.0001:
    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",2")
elif prob <0.001:
    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",1")
else:
    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",0")  
f_flags.close()