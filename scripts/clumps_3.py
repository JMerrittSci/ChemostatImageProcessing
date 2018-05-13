# This code (clump counting) is run on the 'clumps_final' file generated
# by the previous 'clump selection' analysis program. It counts the number
# of 'clumps' found each minute and applies a simple size threshold.
#
# It generates a 'clump_counts' file which represents a time series of
# E. coli cell aggregate abundance.

import numpy as np
import os
import sys
from StringIO import StringIO
from _mysql import NULL

expLabel = sys.argv[1]
sizeThreshold = int(sys.argv[2])

# Open data files
clump_file = open(os.path.join(os.path.join(expLabel,'temp_files'),expLabel+'_clumps_final.csv'),'r')
minute_file = open(os.path.join(expLabel,expLabel+'_counts.csv'),'r')

# Make sure folder exists
path = expLabel
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

minuteClumps=[] # Clumps found in each minute
threshold=sizeThreshold # Size below which clumps are discarded
minutes=np.genfromtxt(minute_file,delimiter=",")

clump_line=clump_file.readline() # Read in a clump line
emptyFile=False
clump= np.genfromtxt(StringIO(clump_line),delimiter=',')
for line in minutes: # For each minute...
    minute = line[0] # Record this minute
    minuteClumps.append([minute,[]]) # Add a list of clumps for this minute
    while clump_line != "" and clump_line != "\n" and len(clump)>0 and clump[0] < minute and not emptyFile: # While this clump was taken from an image before this minute
        clump_line=clump_file.readline() # Read in a new clump
        if clump_line == NULL or len(clump_line)<3:
            emptyFile=True
        else:
            clump= np.genfromtxt(StringIO(clump_line),delimiter=',')
    while clump_line != "" and clump_line != "\n" and len(clump)>0 and clump[0] == minute and not emptyFile: # While this clump was taken from an image during this minute
        if clump[4]>=threshold: # If this clump meets the size threshold
            minuteClumps[-1][1].append(clump[4]) # Save the clump
        clump_line=clump_file.readline() # Then read in a new clump
        if clump_line == NULL or len(clump_line)<3:
            emptyFile=True
        else:
            clump= np.genfromtxt(StringIO(clump_line),delimiter=',')

# Create the 'clumps_count' output file.
f = open(os.path.join(expLabel,expLabel+'_clump_counts.csv'),'w')
for i in range(len(minuteClumps)): # Output the data. Note that we divide by 5 to get an average count for this minute, as each minute contains 5 frames
    f.write(str(minuteClumps[i][0])+","+str(float(len(minuteClumps[i][1]))/5.0)+"\n")
f.close()
