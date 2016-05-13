import numpy as np
import os
import sys
from StringIO import StringIO

expLabel = sys.argv[1]

cell_file = open(os.path.join(os.path.join(expLabel,'datasets'),expLabel+'_dataset.csv'),'r')
clump_file = open(os.path.join(os.path.join(expLabel,'datasets'),expLabel+'_clump_dataset.csv'),'r')

path = os.path.join(expLabel,'temp_files')
datefolder=path
filename = 'clump_corrected'
filetype = '.csv'
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

path = os.path.join(path,expLabel+'_'+filename)
f = open(path+filetype,'w')

cell=cell_file.readline()
cell_data= np.genfromtxt(StringIO(cell),delimiter=',')

minute = -1
frame = -1
frame_cells=[]

for line in clump_file:
    clump = np.genfromtxt(StringIO(line),delimiter=',')
    newMinute=clump[0]
    newFrame=clump[1]
    keepClump = True
    if not (newMinute==minute and newFrame==frame):
        minute = newMinute
        frame = newFrame
        while len(cell)>2 and cell_data[0] < minute or cell_data[1] < frame:
            cell=cell_file.readline()
            if len(cell)>2:
                cell_data= np.genfromtxt(StringIO(cell),delimiter=',')
        frame_cells=[]
        while len(cell)>2 and cell_data[0] == minute and cell_data[1] == frame:
            if cell_data[2]>0:
                frame_cells.append(cell_data)
            cell=cell_file.readline()
            if len(cell)>2:
                cell_data= np.genfromtxt(StringIO(cell),delimiter=',')
            
    clump_cells=[]
    radius=np.sqrt((float(clump[4])/np.pi))/2.0
    for i in frame_cells:
        if np.abs(i[3]-clump[2]) < radius and np.abs(i[4]-clump[3]) < radius:
            clump_cells.append(i)
    if len(clump_cells)>0:
        clump_cells=np.array(clump_cells)
        cells_area=np.sum(clump_cells[:,5])
        if cells_area > clump[4]*0.75:
            keepClump=False
    
    if keepClump:
        f = open(path+filetype,'a')
        f.write(str(clump[0]))
        for i in range(1,len(clump)):
            f.write(','+str(clump[i]))
        f.write('\n')
        f.close()
