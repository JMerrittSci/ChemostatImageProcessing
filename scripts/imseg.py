import numpy as np
from scipy.ndimage.measurements import label

from skimage.feature import blob_dog, blob_log
import skimage.measure as imeas
import skimage.io as io
from skimage.filters import rank
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.filters.rank import median
from skimage import exposure

from sklearn import svm
from sklearn import preprocessing

import os
import sys
from math import sqrt

expLabel = sys.argv[1]
imgDirectory = sys.argv[2]
filePrefix = sys.argv[3]
experimentLength = int(sys.argv[4])
trainingSetFilePath = sys.argv[5]
FPM = 5

data = np.genfromtxt(open(trainingSetFilePath,'r'),delimiter=",")
labels=data[:,0]
scaler=preprocessing.StandardScaler().fit(data[:,1:])
features=scaler.transform(data[:,1:])

clf = svm.SVC(kernel='rbf',C=10.0 ** 5,gamma=10.0 ** (-3))
clf.fit(features, np.ravel(labels))

def Moments(data):
    # From: Gabriel Landini
    # http://fiji.sc/Auto_Threshold#Installation
    # https://github.com/fiji/Auto_Threshold/blob/master/src/main/java/fiji/threshold/Auto_Threshold.java
    
    # W. Tsai, "Moment-preserving thresholding: a new approach," Computer Vision,
    # Graphics, and Image Processing, vol. 29, pp. 377-393, 1985.
    # Ported to ImageJ plugin by G.Landini from the the open source project FOURIER 0.8
    # by  M. Emre Celebi , Department of Computer Science,  Louisiana State University in Shreveport
    # Shreveport, LA 71115, USA
    # http://sourceforge.net/projects/fourier-ipal
    # http://www.lsus.edu/faculty/~ecelebi/fourier.htm 
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

def maskhist(image, coords):
    values=[]
    for i in coords:
        values.append(int(image[i[0],i[1]]))
    hist = np.bincount(np.ravel(values))
    bin_centers = np.arange(len(hist))
    idx = np.nonzero(hist)[0][0]
    return hist[idx:], bin_centers[idx:]

path=expLabel
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


baseName = filePrefix+'-'
filePath =  imgDirectory

data = np.zeros((0,15))
selem = disk(10)
loadedImage=True
                
for i in range(0,experimentLength):
    b_list = []
    for j in range(0,FPM):
        if os.path.isfile(filePath+baseName+str(i).zfill(6)+'-'+str(j).zfill(2)+'.pgm'):            
            loadedImage=True
            try:
                image_gray= io.imread(filePath+baseName+str(i).zfill(6)+'-'+str(j).zfill(2)+'.pgm')
            except:
                loadedImage=False
                  
            if loadedImage:
                image_gray= median(image_gray, disk(1))
                image_gray= rank.mean_bilateral(image_gray,selem=selem, s0=5, s1=5)
                imAvBrightness=np.mean(image_gray)
                imMaxBrightness=np.max(image_gray)
                
                blobs_dog = blob_dog(image_gray,min_sigma = 2, max_sigma=8, threshold=0.02+0.02*imMaxBrightness/255.0,overlap=0.9)
                if len(blobs_dog)>0:
                    blobs_dog[:,2] = blobs_dog[:,2]*sqrt(2)  
                    
                    blobIndex=0
                    blobdata=[]
                    blobpos=[]
                    firstblob=True
                    while blobIndex < len(blobs_dog):
                        y, x, r = blobs_dog[blobIndex]
                        xmin=np.max([0,int(x-2*r)])
                        xmax=np.min([int(x+2*r),len(image_gray[0])])
                        ymin=np.max([0,int(y-2*r)])
                        ymax=np.min([int(y+2*r),len(image_gray)])
                        atTopFrame=(ymin<1)
                        atBotFrame=(ymax>len(image_gray)-2)
                        atLeftFrame=(xmin<1)
                        atRightFrame=(xmax>len(image_gray[0])-2)
                        blob_image=np.copy(image_gray[ymin:ymax,xmin:xmax])
                        masks=[]
                        numMasks=15
                        maskDivisor=numMasks + int(numMasks/2.0)
                        maskMin=np.min(blob_image)
                        maskRange=np.max(blob_image)-maskMin
                        
                        tempratio=float(maskRange)/float(maskDivisor)
                        if tempratio==0:
                            tempratio=1.0    
                        for k in range(maskDivisor-numMasks,maskDivisor):
                                   
                            tempim=binary_dilation(binary_erosion((blob_image-maskMin)/(tempratio)>k,disk(1)),disk(2))*(k-(maskDivisor-numMasks))
                            masks.append(tempim)
                        blob_image_2=np.maximum.reduce(masks)
                        
                        Rlabel=0
                        label_img=np.zeros((len(blob_image_2),len(blob_image_2[0])),dtype=int)
                        for k in range(numMasks):
                            masks[k]=blob_image_2==k
                            labeledmask,Rnum=label(masks[k],structure=np.ones((3,3)))
                            label_img+=Rlabel*masks[k]
                            label_img+=labeledmask
                            Rlabel+=int(Rnum)
                        lnum=Rlabel+1

                        neighbors=np.zeros((lnum,lnum),dtype=bool)
                        brightness_regions=np.zeros((lnum),dtype=int)
                        for m in range(len(blob_image)-1):
                            for n in range(len(blob_image[0])-1):
                                neighbors[label_img[m][n]][label_img[m+1][n]]=True
                                neighbors[label_img[m][n]][label_img[m][n+1]]=True
                                neighbors[label_img[m+1][n]][label_img[m][n]]=True
                                neighbors[label_img[m][n+1]][label_img[m][n]]=True
                        for m in range(len(blob_image)):
                            for n in range(len(blob_image[0])):
                                brightness_regions[label_img[m][n]]=blob_image_2[m][n]
                        
                        props = reversed(imeas.regionprops(label_img))     
                        mask=np.zeros((len(blob_image_2),len(blob_image_2[0])),dtype=int)
                
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

                        regstatus=[[0] for x in range(lnum)]
                        nextLabel=1
                        labelCenters=[]
                        for k in props:
                            if brightness_regions[k.label]==0:
                                regstatus[k.label][0]=-1
                            elif k.label in borders_indices:
                                regstatus[k.label][0]=-2
                            else:
                                seed=False
                                edgeNeighbor=False
                                onlyEdgeNeighbor=True
                                seedID=[]
                                for m in range(0,k.label)+range(k.label+1,lnum):
                                    if neighbors[k.label,m]:
                                        if regstatus[m][0]==-1:
                                            regstatus[k.label]=[-1]
                                            break
                                        elif regstatus[m][0]!=0:
                                            if regstatus[m][0]==-2:
                                                edgeNeighbor=True
                                            else:
                                                onlyEdgeNeighbor=False
                                                seed=True
                                                seedID=list(set(seedID+regstatus[m]))
                                if edgeNeighbor and onlyEdgeNeighbor:
                                    regstatus[k.label]=[-2]
                                elif seed==True:
                                    regstatus[k.label]=seedID
                                else:
                                    regstatus[k.label][0]=nextLabel
                                    labelCenters.append(k.centroid)
                                    nextLabel+=1
                                if regstatus[k.label][0]>0:
                                    if len(regstatus[k.label])==1:
                                        for m in k.coords:
                                            mask[m[0]][m[1]]=regstatus[k.label][0]
                                    else:
                                        for m in k.coords:
                                            mask[m[0]][m[1]]=regstatus[k.label][np.argmin([np.linalg.norm(a-m) for a in [labelCenters[b-1] for b in regstatus[k.label]]])]
        
                        
                        props=imeas.regionprops(mask,blob_image)
                       
                        for k in range(len(props)):
                            if firstblob:
                                blobdata = np.array([props[k].area,props[k].convex_area-props[k].area,props[k].eccentricity,props[k].major_axis_length,props[k].minor_axis_length,
                                                 props[k].max_intensity,props[k].mean_intensity,imAvBrightness,imMaxBrightness,np.var(maskhist(blob_image,props[k].coords))])
                                blobpos = np.array([xmin+props[k].centroid[1],ymin+props[k].centroid[0]])
                                firstblob=False
                            else:
                                blobdata = np.vstack((blobdata,np.array([props[k].area,props[k].convex_area-props[k].area,props[k].eccentricity,props[k].major_axis_length,props[k].minor_axis_length,
                                                 props[k].max_intensity,props[k].mean_intensity,imAvBrightness,imMaxBrightness,np.var(maskhist(blob_image,props[k].coords))])))
                                blobpos = np.vstack((blobpos, np.array([xmin+props[k].centroid[1],ymin+props[k].centroid[0]])))
                        blobIndex+=1
                    if len(blobdata)>0:
                        predictedLabels=clf.predict(scaler.transform(np.copy(blobdata)))
                        if len(blobdata.shape)==1:
                            blobdata=[blobdata]
                            blobpos=[blobpos]
                        f = open(path2+filetype2,'a')
                        for k1 in range(len(blobdata)):
                            f.write(str(i)+','+str(j)+','+str(predictedLabels[k1])+',' + str(blobpos[k1][0])+','+str(blobpos[k1][1])+',')
                            for k2 in range(len(blobdata[0])):
                                f.write(str(blobdata[k1][k2]))
                                if k2<len(blobdata[0])-1:
                                    f.write(',')
                                else:
                                    f.write('\n')
                        f.close()
                        
                hist,bin_centers=exposure.histogram(image_gray)
                threshold=bin_centers[Moments(hist)]
                label_img,num_labels=label(image_gray>threshold)
                props=imeas.regionprops(label_img,image_gray)
                if len(props)>0:
                    f = open(pathClump+filetypeClump,'a')
                    for k in range(len(props)):
                        if props[k].area>500:
                            f.write(str(i)+','+str(j)+','+str(props[k].centroid[1])+','+str(props[k].centroid[0])+','+
                                str(props[k].area)+','+str(props[k].convex_area-props[k].area)+','+str(props[k].eccentricity)+','+
                                str(props[k].max_intensity)+','+str(props[k].mean_intensity)+','+str(np.var(maskhist(image_gray,props[k].coords)))+'\n')
                    f.close()
                            
