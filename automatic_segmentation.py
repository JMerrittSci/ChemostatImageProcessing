import subprocess
import datetime

########################################################################
###                   USER DEFINED SETTINGS                          ###
########################################################################

expLabel= 'Example_Experiment' # A descriptive label for the experiment.
    # Determines the name of the folders and files generated. If
    # left blank, defaults to the date this script was run.
imgDirectory= 'test_imgs/' # The directory containing the images.
filePrefix= '2015-06-24' # The image filename prefix.
    # Note that where filePrefix is PREFIX, the assumed filename
    # convention is PREFIX-aaaaaa-bb.pgm, where "aaaaaa" is the
    # 6-digit minute number the image was taken during and
    # "bb" is the image's 2-digit frame number (note that the
    # code currently requires there to be 5 frames per minute). The
    # first image of an experiment might be PREFIX-000000-00.pgm
    # (the test images included are from later time points).
firstMinute=5500 # Usually would be 0; note that this corresponds
    # to the lowest value of "aaaaaa" of any image file according
    # to the convention defined above.
experimentLength=5510 # Usually the number of minutes in the
    # experiment; note that this corresponds to the highest
    # value of "aaaaaa" of any image file according to the
    # convention defined above.
trainingSetFilePath='scripts/chromosomal.csv' # The filepath of
    # the machine learning training set for this strain and
    # experimental condition.
clumpBrightnessThreshold=150 # A clump must have a max pixel
    # intensity (after smoothing) of at least this value or be
    # discarded. The intensity range is 0-255.
clumpMeanBrightnessThreshold=100 # A clump must have a mean pixel
    # intensity (after smoothing) of at least this value or be
    # discarded. The intensity range is 0-255.
clumpSizeThreshold=500 # A clump must have a pixel value area
    # at least this high or be discarded. Note that the minimum
    # clump size is already 500 pixels internally; clumps must
    # have an area of at least 500 pixels to be counted
    # regardless of this threshold value.

########################################################################
########################################################################

if len(expLabel)==0:
    expLabel=str(datetime.date.today())
print 'Working on images from experiment "'+expLabel+'"... (this will take a while)'
process = subprocess.Popen(['python', 'scripts/imseg.py',expLabel,imgDirectory,filePrefix,str(experimentLength),trainingSetFilePath])
subprocess.Popen.wait(process)
print "Image segmentation: done! [Step 1/7]"
process = subprocess.Popen(['python', 'scripts/cells_1.py',expLabel,str(firstMinute)])
subprocess.Popen.wait(process)
print "Spatial filtering: done! [Step 2/7]"
process = subprocess.Popen(['python', 'scripts/cells_2.py',expLabel,str(firstMinute)])
subprocess.Popen.wait(process)
print "Frame variance filtering: done! [Step 3/7]"
process = subprocess.Popen(['python', 'scripts/cells_3.py',expLabel])
subprocess.Popen.wait(process)
print "Cell time series generation: done! [Step 4/7]"
process = subprocess.Popen(['python', 'scripts/clumps_1.py',expLabel])
subprocess.Popen.wait(process)
print "Cell-clump comparison: done! [Step 5/7]"
process = subprocess.Popen(['python', 'scripts/clumps_2.py',expLabel,str(clumpBrightnessThreshold),str(clumpMeanBrightnessThreshold)])
subprocess.Popen.wait(process)
print "Clump brightness thresholding: done! [Step 6/7]"
process = subprocess.Popen(['python', 'scripts/clumps_3.py',expLabel,str(clumpSizeThreshold)])
subprocess.Popen.wait(process)
print "Clump time series generation: done! [Step 7/7]"
print 'Time series for experiment "'+expLabel+'" DONE!'
