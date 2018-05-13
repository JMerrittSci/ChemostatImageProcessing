import numpy as np
from math import sqrt

import skimage.measure as imeas
import skimage.io as io
from skimage.feature import blob_dog
from skimage.filter import rank
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.filter.rank import median
from skimage import exposure

from sklearn import svm
from sklearn import preprocessing

from scipy.ndimage.measurements import label

import os
import sys

expLabel = sys.argv[1]
imgDirectory = sys.argv[2]
filePrefix = sys.argv[3]
experimentLength = int(sys.argv[4])
trainingSetFilePath = sys.argv[5]
FPM = 5

# Open hand-labeled training set
data = np.genfromtxt(open(trainingSetFilePath,'r'),delimiter=",")
labels=data[:,0] # Object labels are in first column
scaler=preprocessing.StandardScaler().fit(data[:,1:]) # Compute mean and variance (for future transformation)
features=scaler.transform(data[:,1:]) # Use scaler to set training mean to 0/variance to 1

# Set up basic SVM with RBF kernel. C, gamma values chosen to maximize accuracy as determined by cross-validation.
clf = svm.SVC(kernel='rbf',C=10.0 ** 5,gamma=10.0 ** (-3))
clf.fit(features, np.ravel(labels)) # Train SVM using training set.

def maskhist(image, coords):
    # The histogram function from skimage.exposure, modified to only
    # consider specific pixels from an image. Returns histogram of
    # pixel values (values of histogram bins followed by bin centers)
    
    values=[]
    for i in coords:
        values.append(int(image[i[0],i[1]]))
    hist = np.bincount(np.ravel(values))
    bin_centers = np.arange(len(hist))

    # clip histogram to start with a non-zero bin
    idx = np.nonzero(hist)[0][0]
    return hist[idx:], bin_centers[idx:]

def Moments(data): # Returns a threshold for grayscale images
    # From: Gabriel Landini
    #     http://fiji.sc/Auto_Threshold#Installation
    #     https://github.com/fiji/Auto_Threshold/blob/master/src/main/java/fiji/threshold/Auto_Threshold.java
    #    
    #     W. Tsai, "Moment-preserving thresholding: a new approach," Computer Vision,
    #     Graphics, and Image Processing, vol. 29, pp. 377-393, 1985.
    #     Ported to ImageJ plugin by G.Landini from the the open source project FOURIER 0.8
    #     by  M. Emre Celebi , Department of Computer Science,  Louisiana State University in Shreveport
    #     Shreveport, LA 71115, USA
    #     http://sourceforge.net/projects/fourier-ipal
    #     http://www.lsus.edu/faculty/~ecelebi/fourier.htm
    total=0.0
    m0=1.0
    m1=0.0
    m2=0.0
    m3=0.0
    tempsum=0.0
    p0=0.0
    cd=0.0
    c0=0.0
    c1=0.0
    z0=0.0
    z1=0.0
    threshold=-1
    
    histo = [0.0 for i in range(len(data))] #normalized histogram
    
    for i in range(len(data)):
        total+=data[i]
    
    for i in range(len(data)):
        histo[i]=float(data[i]/total)

    # Calculate the first, second, and third order moments
    for i in range(len(data)):
        m1 += i*histo[i]
        m2 += i*i*histo[i]
        m3 += i*i*i*histo[i]
    
    # First 4 moments of the gray-level image should match the first 4 moments
    # of the target binary image. This leads to 4 equalities whose solutions 
    # are given in the Appendix of Ref. 1 
    
    cd = m0 * m2 - m1 * m1
    if cd != 0:
        c0 = (-m2 * m2 + m1 * m3) / cd
        c1 = (m0 * -m3 + m2 * m1) / cd
        z0 = 0.5 * (-c1 - np.sqrt(c1 * c1 - 4.0 * c0))
        z1 = 0.5 * (-c1 + np.sqrt(c1 * c1 - 4.0 * c0))
        if (z1-z0) != 0:
            p0 = (z1 - m1) / (z1 - z0) # Fraction of the object pixels in the target binary image 
        
            # The threshold is the gray-level closest  
            # to the p0-tile of the normalized histogram 
            tempsum=0
            for i in range(len(data)):
                tempsum+=histo[i]
                if tempsum>p0:
                    threshold=i
                    break
        else:
            threshold=0
    else:
        threshold=0
    return threshold

# Following blocks of code create output files
path = expLabel
try: 
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise
    
path2 = os.path.join(expLabel,'datasets')
filename2 = 'dataset'
filetype2 = '.csv'
try: 
    os.makedirs(path2)
except OSError:
    if not os.path.isdir(path2):
        raise
path2 = os.path.join(path2,expLabel+'_' + filename2)
if os.path.isfile(path2 + filetype2):
    testNum = 0
    while os.path.isfile(path2 + '_' + str(testNum) + filetype2):
        testNum+=1
    path2=path2+'_' + str(testNum)
    
f = open(path2+filetype2,'w')
f.write('')
f.close()

pathClump = os.path.join(expLabel,'datasets')
filenameClump = 'clump_dataset'
filetypeClump = '.csv'
try: 
    os.makedirs(pathClump)
except OSError:
    if not os.path.isdir(pathClump):
        raise
pathClump = os.path.join(pathClump,expLabel+'_' + filenameClump)
if os.path.isfile(pathClump + filetypeClump):
    testNum = 0
    while os.path.isfile(pathClump + '_' + str(testNum) + filetypeClump):
        testNum+=1
    pathClump=pathClump+'_' + str(testNum)
    
f = open(pathClump+filetypeClump,'w')
f.write('')
f.close()

# Set filepath and filename prefix for images
baseName = filePrefix+'-'
filePath =  imgDirectory

disk10 = disk(10) # A disk-structuring image with a radius of 10 pixels
loadedImage=True # Variable tracks if image was successfully loaded

for i in range(0,experimentLength):
    frameCounts=[] # Estimate of number of objects detected per 'frame' (image taken this minute)
    for j in range(0,FPM): # Iterate over 5 images taken this minute
        if os.path.isfile(filePath+baseName+str(i).zfill(6)+'-'+str(j).zfill(2)+'.pgm'): # (Do nothing otherwise)
            loadedImage=True # Be optimistic
            try:
                image_gray= io.imread(filePath+baseName+str(i).zfill(6)+'-'+str(j).zfill(2)+'.pgm') # Load the image!
            except:
                loadedImage=False
                  
            if loadedImage: # (Do nothing otherwise)
                # We run a median filter over the image to eliminate both isolated 'hot' and 'cold' pixels,
                # due to the camera itself or random noise, without actually blurring the image. This adds
                # slightly unpredictable distortion to the image, but because we only consider a tiny local
                # neighborhood of each pixel, in practice it doesn't matter.
                image_gray=median(image_gray, disk(1))
                
                # The bilateral mean is effectively a selective Gaussian blur,
                # smoothing the image without mixing across edges of substantially
                # different structures.
                image_gray= rank.mean_bilateral(image_gray,selem=disk10, s0=5, s1=5)
                
                # Average and max brightness across entire image...
                imAvBrightness=np.mean(image_gray)
                imMaxBrightness=np.max(image_gray)
                            
                # Use DoG method to find 'blobs' (hypothetical cells in general size range of cells)
                # in image, using slightly dynamic thresholding based on overall image brightness
                blobs_dog = blob_dog(image_gray,min_sigma = 2, max_sigma=8, threshold=0.02+0.02*imMaxBrightness/255.0,overlap=0.9)
                if len(blobs_dog)>0: # Proceed with cell analysis if we see any blobs
                    blobs_dog[:,2]=blobs_dog[:,2]*sqrt(2) # Convert column to approximate radius of blob 
                    
                    blobdata=[] # Various data about blobs
                    blobpos=[] # Positions of blobs
                    sentBack=False # Part of control code for labeling training sets; irrelevant
                    firstblob=True # Because the first blob forms the first row in a Numpy array
                    
                    blobIndex=0
                    while blobIndex < len(blobs_dog): # While chosen instead of for loop for labeling training sets...
                        y, x, r = blobs_dog[blobIndex]  # Position and radius of blob in image. The coordinate
                                                        # corresponding to 'x' or 'y' will swap throughout this
                                                        # code depending on the method...
                        
                        # Generate borders for a sub-image around the blob, going twice the blob's radius in each direction (without leaving the borders of the image)
                        xmin=np.max([0,int(x-2*r)])
                        xmax=np.min([int(x+2*r),len(image_gray[0])])
                        ymin=np.max([0,int(y-2*r)])
                        ymax=np.min([int(y+2*r),len(image_gray)])
                        
                        # In theory objects at the edge of the image should be treated differently, as they are generally not picked up by the DoG pass
                        atTopFrame=(ymin<1)
                        atBotFrame=(ymax>len(image_gray)-2)
                        atLeftFrame=(xmin<1)
                        atRightFrame=(xmax>len(image_gray[0])-2)
                        
                        # Generate the sub-image
                        blob_image=np.copy(image_gray[ymin:ymax,xmin:xmax])
                        
                        masks=[] # Prepare an array for the 15 binary masks we'll be making
                        numMasks=15
                        maskDivisor=numMasks + int(numMasks/2.0) # Read: (3/2)*numMasks
                        maskMin=np.min(blob_image) # Minimum value of image
                        maskRange=np.max(blob_image)-maskMin # Range of pixel values of entire image

                        tempratio=float(maskRange)/float(maskDivisor)   # Read: (2/3)*(Range of pixel values)/(number of masks).
                                                                        # View this as defining bin widths for bins equal to the
                                                                        # number of masks we'll make, but only over 2/3rds the
                                                                        # intensity range of our actual image (the lower 1/3rd will
                                                                        # be discarded).
                        if tempratio==0: # This is a perfectly uniform image. Shouldn't be possible if we saw a blob here, but...
                            tempratio=1.0 # Just to make sure the code doesn't crash when we divide, although nothing interesting is going to happen in a uniform image.
                        for k in range(maskDivisor-numMasks,maskDivisor): # Read: range(0.5*numMasks, 1.5*numMasks)
                            # The following line needs explanation. "(blob_image-maskMin)/(tempratio)>k" generates a binary mask showing
                            # all pixels with intensity values AT LEAST high enough to be in the range of the 'bins' described above
                            # Due to the values of 'k' chosen the darkest 1/3rd of pixels are automatically discarded, with the masks
                            # actually representing ranges over the top 2/3rds of pixel intensities only.
                            #
                            # The binary erosion removes isolated features from the masks (we want masks to represent actual features
                            # of the image), and the dilation tries to restore the original size with the negative side effect of
                            # making all features more circular.
                            #
                            # The final step multiplies our mask (pixel values 0 and 1) by "(k-(maskDivisor-numMasks)" which will be a number
                            # between 0 and 14. Note that this means the darkest mask is multiplied by 0 and therefore irrelevant! This is
                            # technically a bug but all analysis was run with the bug intact, meaning it cannot be fixed without invalidating
                            # comparisons to existing data sets. In practice, this bug doesn't matter; the program still works, it just uses
                            # the top ~62% of pixel intensities instead of the top ~67% in doing calculations.
                            tempim=binary_dilation(binary_erosion((blob_image-maskMin)/(tempratio)>k,disk(1)),disk(2))*(k-(maskDivisor-numMasks))
                            masks.append(tempim) # Save the mask
                        blob_image_2=np.maximum.reduce(masks)   # Layer the masks, keeping the largest mask value in place for each pixel
                                                                # In theory this produces a 16-color simplified image, but due to the above
                                                                # bug it will be only 15-color, with pixel values ranging from 0-14.
                        
                        # We now want to chop the layered mask image up into distinct labeled regions.
                        Rlabel=0 # Number of regions - will be updated and used WHILE regions are added!
                        label_img=np.zeros((len(blob_image_2),len(blob_image_2[0])),dtype=int)  # Make a blank image which will be filled with
                                                                                                # labeled regions from ALL masks!
                        for k in range(numMasks): # We're going to overwrite the original 15 masks...
                            masks[k]=blob_image_2==k    # Make a binary mask of ONLY where the layered mask image has an intensity matching
                                                        # the range ('bin,' as described above) this mask was intended to cover.
                            # We now generate a labeled version of the above mask, labeling distinct (non-contiguous) regions with
                            # separate identifiers. Note that to get the desired output we use the SciPy region labeling method,
                            # NOT the scikit-image version!
                            labeledmask,Rnum=label(masks[k],structure=np.ones((3,3))) # Created the labeled mask (Rnum = number of regions)
                            label_img+=Rlabel*masks[k]  # First multiply the binary mask by Rlabel - the value of the next label in the overall
                                                        # labeled image - and add the mask to the overall labeled image.
                            label_img+=labeledmask # Now add the labeled mask to the overall labeled image, which now gives each region a distinct, final label
                            Rlabel+=int(Rnum) # Increment Rlabel by the number of regions just added
                        lnum=Rlabel+1 # Actual number of labels, counting "0"
                     
                        neighbors=np.zeros((lnum,lnum),dtype=bool)  # An (initially false) boolean matrix showing which regions
                                                                    # directly neighbor other regions
                        
                        # Iterate through the image, updating the array of neighbors to show which regions are connected to each other
                        for m in range(len(blob_image)-1):
                            for n in range(len(blob_image[0])-1):
                                neighbors[label_img[m][n]][label_img[m+1][n]]=True
                                neighbors[label_img[m][n]][label_img[m][n+1]]=True
                                neighbors[label_img[m+1][n]][label_img[m][n]]=True
                                neighbors[label_img[m][n+1]][label_img[m][n]]=True
                                
                        # Record the actual brightness (mask value, 0-14) of each region
                        brightness_regions=np.zeros((lnum),dtype=int)
                        for m in range(len(blob_image)):
                            for n in range(len(blob_image[0])):
                                brightness_regions[label_img[m][n]]=blob_image_2[m][n]
                        
                        # The first time we call regionprops to get information about labeled
                        # regions in our image. In this case, we only use it as an easy way to
                        # get the centers of each labeled region and the collection of pixels
                        # which make up the region.
                        props = reversed(imeas.regionprops(label_img))   
                        
                        # We generate (yet another) labeled mask, initially blank. In this case,
                        # the goal is to generate a labeled mask of separate BLOBS (potential cells)
                        # within the image, rather than labeled regions of similar brightness.
                        mask=np.zeros((len(blob_image_2),len(blob_image_2[0])),dtype=int)
                        
                        # If our sub-image (the one we've been working on for some time now) was
                        # at the edge of the original image from the microscope, we ultimately give
                        # preference to objects originating from that edge. The reason is that while
                        # such objects would likely not otherwise be counted, objects in the original
                        # image that here are originating at the edge of the sub-image are likely
                        # to be counted by a separate hit from the DoG pass, and are discarded here
                        # to prevent double-counting.
                        borders = np.zeros_like(label_img, dtype=np.bool_)
                        if not atLeftFrame:
                            borders[:,0] = True
                        if not atRightFrame:
                            borders[:,len(borders[0])-1] = True
                        if not atTopFrame:
                            borders[0,:] = True
                        if not atBotFrame:
                            borders[len(borders)-1,] = True
                        borders_indices = np.unique(label_img[borders])
                       
                        # The following code is modeled after Lindeberg's watershed-based
                        # blob detection algorithm. Essentially, regions of local maxima are declared
                        # 'seeds' for blobs, which expand outward by absorbing neighboring darker regions
                        regstatus=[[0] for x in range(lnum)]    # The region status of each region has three possible states:
                                                                # 1) [-1], corresponding to blobs on the edge of the sub-image (treated as background, as they should be captured by a separate event from the DoG pass)
                                                                # 2) [0], corresponding to a region with default/unassigned status
                                                                # 3) A list of positive integers, corresponding to IDs of blobs the region is part of or split between
                        nextLabel=1 # Label of the next seed blob to be defined
                        labelCenters=[] # Centers of initial seed regions
                        for k in props: # Iterate over labeled regions. It is important to note that regions here are ordered, with brightest regions appearing first...
                            if k.label in borders_indices: # If this region is on the edge of the sub-image...
                                regstatus[k.label]=[-1] # Declare this region's status to be as such (to be treated as background)
                            elif brightness_regions[k.label]>0: # If this region has non-zero brightness (i.e., is not absolute background)
                                blobNeighbor=False # Assume this region doesn't neighbor any blobs
                                edgeNeighbor=False # Assume this region doesn't neighbor any edge regions
                                seedIDs=[] # List of blobs this region neighbors (by seed ID)
                                for m in range(0,k.label)+range(k.label+1,lnum): # Iterate over all regions (except this one)
                                    if neighbors[k.label,m]: # If I neighbor this region...
                                        if regstatus[m][0]==-1: # And it's an edge region...
                                            edgeNeighbor=True # I neighbor an edge region
                                        elif regstatus[m][0]>0: # Or, if it's part of a growing blob...
                                            blobNeighbor=True # I neighbor a blob...
                                            seedIDs=list(set(seedIDs+regstatus[m])) # And I add that blob's ID to my list of neighboring blobs
                                if edgeNeighbor and not blobNeighbor: # If I only neighbor edge regions (and background/regions darker than myself)...
                                    regstatus[k.label]=[-1] # Label myself as an edge region
                                elif blobNeighbor==True: # Otherwise, if I neighbor a growing blob (meaning I'm not a local maxima)
                                    regstatus[k.label]=seedIDs # Record the blobs I neighbor
                                    for m in k.coords: # Competitively assign my pixels to the blobs, by which seed region centroid is nearest
                                        mask[m[0]][m[1]]=seedIDs[np.argmin([np.linalg.norm(a-m) for a in [labelCenters[b-1] for b in seedIDs]])]
                                else: # Otherwise, I'm a local maxima and therefore a new seed...
                                    regstatus[k.label]=[nextLabel] # Assign my status to my new seed ID
                                    labelCenters.append(k.centroid) # Add my center to the labelCenters
                                    for m in k.coords:
                                        mask[m[0]][m[1]]=nextLabel # Assign all my pixels to myself
                                    nextLabel+=1 # Increment next seed ID
                        
                        # Now that we have parsed the subimage into distinct blobs, we analyze the mask showing
                        # the distinct blobs to get region properties, with reference to the original subimage
                        # gray values.
                        props=imeas.regionprops(mask,blob_image)
                       
                        if sentBack and len(props)==0: # Control code for generating training sets (not relevant here, sentBack is always false)
                            if blobIndex>0:
                                blobIndex-=1
                            else:
                                sentBack=False
                                blobIndex+=1
                        else:
                            if sentBack: # Irrelevant here; sentBack is always False
                                for k in props:
                                    blobdata.pop()
                                    blobpos.pop()
                                sentBack=False
                            for k in range(len(props)):
                                if firstblob: # If this is the first blob detected in the entire image...
                                    # ...generate new arrays for blob properties and position.
                                    blobdata = np.array([props[k].area,props[k].convex_area-props[k].area,props[k].eccentricity,props[k].major_axis_length,props[k].minor_axis_length,
                                                     props[k].max_intensity,props[k].mean_intensity,imAvBrightness,imMaxBrightness,np.var(maskhist(blob_image,props[k].coords))])
                                    blobpos = np.array([xmin+props[k].centroid[1],ymin+props[k].centroid[0]])
                                    firstblob=False
                                else: # Otherwise, append the new blob's properties and position to the arrays previously created for this image.
                                    blobdata = np.vstack((blobdata,np.array([props[k].area,props[k].convex_area-props[k].area,props[k].eccentricity,props[k].major_axis_length,props[k].minor_axis_length,
                                                     props[k].max_intensity,props[k].mean_intensity,imAvBrightness,imMaxBrightness,np.var(maskhist(blob_image,props[k].coords))])))
                                    blobpos = np.vstack((blobpos, np.array([xmin+props[k].centroid[1],ymin+props[k].centroid[0]])))
                            blobIndex+=1
                            
                    if len(blobdata)>0: # If we detected blobs in the image after all that...
                        predictedLabels=clf.predict(scaler.transform(np.copy(blobdata))) # Transform a copy of the blob data, and predict labels with the SVM
                        if len(blobdata.shape)==1: # If only one blob was detected, we have a 1D array; this forces is to be a 2D array with one row
                            blobdata=[blobdata]
                            blobpos=[blobpos]
                        f = open(path2+filetype2,'a')   # Output minute and frame data, predicted label, blob position and blob data.
                                                        # Note that we output in 'append' mode and then close the file to ensure data is saved in case the program is stopped.
                        for k1 in range(len(blobdata)):
                            f.write(str(i)+','+str(j)+','+str(predictedLabels[k1])+',' + str(blobpos[k1][0])+','+str(blobpos[k1][1])+',')
                            for k2 in range(len(blobdata[0])):
                                f.write(str(blobdata[k1][k2]))
                                if k2<len(blobdata[0])-1:
                                    f.write(',')
                                else:
                                    f.write('\n')
                        f.close()
                        
                        frameCounts.append(int(np.sum(predictedLabels))) # Keep an estimate of how many cells were found in this image.
                    
                # Next we try to detect large objects (cell aggregates, which are always bright)
                hist,bin_centers=exposure.histogram(image_gray) # Create a histogram of pixel values over the image
                threshold=bin_centers[Moments(hist)] # Generate a Moments threshold
                label_img,num_labels=label(image_gray>threshold) # Create a labeled mask image of distinct image regions above the threshold
                props=imeas.regionprops(label_img,image_gray) # Calculate properties for these regions
                ncl=0 # Number of 'clumps' (cell aggregates) = 0
                if len(props)>0: # If any clumps were found...
                    f = open(pathClump+filetypeClump,'a') # Output clump data, similar to above code for writing out cell data
                    for k in range(len(props)):
                        if props[k].area>500: # We only record objects at least 500 pixels in area
                            ncl+=1
                            f.write(str(i)+','+str(j)+','+str(props[k].centroid[1])+','+str(props[k].centroid[0])+','+
                                str(props[k].area)+','+str(props[k].convex_area-props[k].area)+','+str(props[k].eccentricity)+','+
                                str(props[k].max_intensity)+','+str(props[k].mean_intensity)+','+str(np.var(maskhist(image_gray,props[k].coords)))+'\n')
                    f.close()
        