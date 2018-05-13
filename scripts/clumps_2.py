# This code (clump selection) is run on the 'clump_corrected' file generated
# by the previous 'clump checker' analysis program. It is designed to apply
# a simple brightness filter over the 'clump_corrected' dataset,
# applied only at this stage (instead of during the 'clump checker' program) in
# to avoid the large computation time from the previous program just to check the
# result of a different brightness filter.
# 
# This code also relies on the existence of a curated 'counts_mod' file, curated
# by hand, to discard clumps from minutes discarded from the cell time series.
#
# This code outputs a 'clumps_final' file which contains a list of all objects
# finally determined to be clumps, from minutes still under consideration.

import numpy as np
import os
import sys
from StringIO import StringIO

expLabel = sys.argv[1]
maxThreshold= int(sys.argv[2])
meanThreshold=int(sys.argv[3])

# Open data files
clump_file = open(os.path.join(os.path.join(expLabel,'temp_files'),expLabel+'_clump_corrected.csv'),'r')
minute_file = open(os.path.join(expLabel,expLabel+'_counts.csv'),'r')

# Following blocks of code create output file
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

# Read in a line from the clump dataset
clump_line=clump_file.readline()
clump= np.genfromtxt(StringIO(clump_line),delimiter=',') # Parse data from clump line
for line in minute_file: # For each line in the minute file (i.e., each 'good' minute)
    minute = np.genfromtxt(StringIO(line),delimiter=',')[0] # Record the minute
    while clump != "" and clump != "\n" and (int(clump[0]) < int(minute)): # While the clump is from an image taken before this minute...
        clump_line=clump_file.readline() # Read in a new clump
        clump= np.genfromtxt(StringIO(clump_line),delimiter=',')
    while clump != "" and clump != "\n" and (int(clump[0]) == int(minute)): # While the clump is from the same image as this minute...
        keepClump=True # Assume we keep the clump
        
        if clump[7]<maxThreshold or clump[8]<meanThreshold: # If the clump fails to meet either threshold (note: clump[7] is maximum intensity, clump[8] is average intensity)
            keepClump=False # Discard the clump
        
        if keepClump: # If we keep the clump...
            f = open(path+filetype,'a')
            f.write(str(clump[0])) # Write it to the output file
            for i in range(1,len(clump)):
                f.write(','+str(clump[i]))
            f.write('\n')
            f.close()
        clump_line=clump_file.readline() # Read in a new clump
        clump= np.genfromtxt(StringIO(clump_line),delimiter=',')