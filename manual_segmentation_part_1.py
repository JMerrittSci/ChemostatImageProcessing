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

########################################################################
########################################################################

if len(expLabel)==0:
    expLabel=str(datetime.date.today())
print 'Working on images from experiment "'+expLabel+'"... (this will take a while)'
process = subprocess.Popen(['python', 'scripts/imseg.py',expLabel,imgDirectory,filePrefix,str(experimentLength),trainingSetFilePath])
subprocess.Popen.wait(process)
print "Image segmentation: done! [Step 1/5]"
process = subprocess.Popen(['python', 'scripts/cells_1.py',expLabel,str(firstMinute)])
subprocess.Popen.wait(process)
print "Spatial filtering: done! [Step 2/5]"
process = subprocess.Popen(['python', 'scripts/cells_2.py',expLabel,str(firstMinute)])
subprocess.Popen.wait(process)
print "Frame variance filtering: done! [Step 3/5]"
process = subprocess.Popen(['python', 'scripts/cells_3.py',expLabel])
subprocess.Popen.wait(process)
print "Cell time series generation: done! [Step 4/5]"
process = subprocess.Popen(['python', 'scripts/clumps_1.py',expLabel])
subprocess.Popen.wait(process)
print "Cell-clump comparison: done! [Step 5/5]"
print 'Cell time series for experiment "'+expLabel+'" DONE!'
print "Perform any necessary work on " + expLabel + "/" + expLabel + "_counts.csv,"
print "then change settings in and run manual_segmentation_part_2.py."
