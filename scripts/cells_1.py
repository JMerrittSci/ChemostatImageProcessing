# This code (spatial distribution labeler) is run on (E.coli) cell data
# sets following completion of image segmentation and SVM labeling of images.
#
# This filtering step determines the 'quality' of each image as a function
# of its likelihood based on the distributions of cell positions, assuming
# cells should be uniformly distributed across the image. Typically this can
# only be thrown off to a significant extent by huge regions with no cells,
# which are nearly always the result of bubbles or large cell aggregates.
#
# This code only labels the 'quality' of each image; it does not filter
# images out. This information is used in the next analysis program,
# the 'minute variance labeler'. This code outputs several files,
# the most important of which is the frame counts flags file.

import sys
import os
import numpy as np
import math
from StringIO import StringIO

expLabel = sys.argv[1]
firstMinute = int(sys.argv[2])

path = os.path.join(expLabel,'temp_files')
datefolder=path
filename0 = 'frame_counts_flags' # File containing number of cells found for each minute and frame, with an estimate of the likelihood of seeing an image with this spatial distribution
filetype = '.csv'
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

# Following blocks of code create output files
path = os.path.join(path,expLabel+'_' + filename0)
if os.path.isfile(path + filetype):
    testNum = 0
    while os.path.isfile(path + '_' + str(testNum) + filetype):
        testNum+=1
    path=path+'_' + str(testNum)
f_flags = open(path+filetype,'w')

# Open cell data file. The file is generally gigabytes large and therefore
# is not loaded into memory all at once, but rather line by line.
data_file = open(os.path.join(os.path.join(expLabel,'datasets'),expLabel+'_dataset.csv'),'r')

endOfFile=False # We have not reached the end of the file.
x_hist=np.zeros(318) # Initialize x and y histograms.
y_hist=np.zeros(237)

minute=firstMinute # Initial minute
frame=0 # Initial frame

counts=0    # Number of cells (using predicted labels)
ones=0      # Number of cells (evaluating all detected cell-objects
            # to 1 cell, regardless of numerical prediction from SVM
       
x=[] # x-locations of cells in image
y=[] # y-locations of cells in image

while not endOfFile: # Until we've reached the end of the file...
    try:
        line=data_file.readline() # Read in a line.
        if line is None: # Break if line is empty
            endOfFile=True
            break
        line=np.genfromtxt(StringIO(line), delimiter=",") # Convert the line to a Numpy array
        if len(line)==0: # Break if Numpy array has no elements
            endOfFile=True
            break
    except: # Break if there's an exception
        endOfFile=True 
        break
    # Note: each line of the data file represents a cell,
    #       with the following properties:
    # line[0] = minute of object detection (data file ordered by this column)
    # line[1] = frame/image number (in this minute) of object detection (secondary ordering of data file by this column)
    # line[2] = SVM prediction of number of cells in object; generally 0, 1, or 2; aggregates handled elsewhere
    # line[3] = x-position of object
    # line[4] = y-position of object
    # later columns have to do with object properties; relevant to SVM but not here
    if minute == int(line[0]+0.01) and frame == int(line[1]+0.01): # If the minute/frame we're considering matches the minute/frame of the object...
        label=int(line[2]+0.01) # Note the predicted label/number of cells
        if label > 0: # If this is predicted to have at least 1 cell...
            x_pos=int(line[3]+0.01) # Save the x and y positions of the object
            y_pos=int(line[4]+0.01)
            if x_pos > 7 and x_pos < 1280 and y_pos > 7 and y_pos < 956: # Reject objects found at edges of image (they throw off the SVM)
                counts+=label # Estimate: found X more cells
                ones+=1 # Estimate: found one more cell-containing object
                for loopVar in range(label): # X times...
                    x.append(x_pos) # Add x and y positions of objects to lists
                    y.append(y_pos)
                    x_hist[x_pos/4-2]+=1 # Add counts to x and y position histograms
                    y_hist[y_pos/4-2]+=1
    else: # If minute/frame we're considering doesn't match that of object we're looking at...
        while minute < int(line[0]+0.01) or (minute == int(line[0]+0.01) and frame < int(line[1]+0.01)): # Until we've 'caught up' to the minute/frame of the object...
            if len(x)==0: # If there are no detected objects for the minute/frame we were considering...
                f_flags.write(str(minute)+","+str(frame)+",0,0\n") # Output that there were no cells.
            else:
                x_mean=np.mean(x) # Average x-location of cells
                y_mean=np.mean(y) # Average y-location of cells
                # Given the size of the image, assuming uniform distributions in both
                # the x and y dimensions, what are the standard deviations of the means
                # in both dimensions, given the number of counts?
                stdevX=math.sqrt((math.pow(1272,2)-1)/12)/math.sqrt(counts) 
                stdevY=math.sqrt((math.pow(948,2)-1)/12)/math.sqrt(counts)
                # How far are the actual means from the center of the image?
                dispX=math.fabs(x_mean-643.5)
                dispY=math.fabs(y_mean-481.5)
                # What is the probability of seeing a mean at least as far from the
                # center of the image, in both dimensions, as we actually see, given
                # the standard error of the mean?
                probX=1-math.erf(dispX/(stdevX*math.sqrt(2)))
                probY=1-math.erf(dispY/(stdevY*math.sqrt(2)))
                # Assuming x and y are uncorrelated (they aren't, but in an ideal image
                # or perfect uniform distribution THEY SHOULD BE, what are the odds
                # of detecting an image with means distributed 'at least this extreme'?
                prob=probX*probY
                if prob < 0.000001: # 1 in 1 million images should be this extreme: label 4
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",4\n")
                elif prob <0.00001: # 1 in 100 thousand images should be this extreme: label 3
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",3\n")
                elif prob <0.0001: # 1 in 10 thousand images should be this extreme: label 2
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",2\n")
                elif prob <0.001: # 1 in 1 thousand images should be this extreme: label 1
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",1\n")
                else: # More than 1 in 1 thousand images should be this extreme: label 0
                    f_flags.write(str(minute)+","+str(frame)+","+str(counts)+",0\n")
            ####
            frame+=1 # increment frame number
            if frame>4: # if the frame number is higher than the total number of frames per image...
                frame=0 # Reset frame number to zero and increase minute number
                minute+=1
            counts=0 # Reset counts/distribution data for next frame
            ones=0
            x=[]
            y=[]
            
        # At this point, the minute and frame match (after previously not having matched).
        # This code is a copy of what would have been done if the minutes and frames did match;
        # see the "if-then" code in the parent if statement, as this is all a repeat.
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
                    x_hist[x_pos/4-2]+=1
                    y_hist[y_pos/4-2]+=1

# When we've run out of cells, finish the analysis on the last frame.
# This code is a copy. See above for more details.
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