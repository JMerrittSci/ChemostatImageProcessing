import subprocess
import datetime

########################################################################
###                   USER DEFINED SETTINGS                          ###
########################################################################

expLabel= 'Example_Experiment' # A descriptive label for the experiment.
    ##############################################################
    # NOTE: MUST MATCH SETTING FROM MANUAL_SEGMENTATION_PART_1.PY
    ##############################################################
    # Determines the name of the folders and files generated. If
    # left blank, defaults to the date this script was run.
clumpBrightnessThreshold=150 # A clump must have a max pixel
    # intensity (after smoothing) of at least this value or be
    # discarded. The intensity range is 0-255.
clumpMeanBrightnessThreshold=150 # A clump must have a mean pixel
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
print 'Finishing up clump analysis for experiment "'+expLabel+'"...'
process = subprocess.Popen(['python', 'scripts/clumps_2.py',expLabel,str(clumpBrightnessThreshold),str(clumpMeanBrightnessThreshold)])
subprocess.Popen.wait(process)
print "Clump brightness thresholding: done! [Step 1/2]"
process = subprocess.Popen(['python', 'scripts/clumps_3.py',expLabel,str(clumpSizeThreshold)])
subprocess.Popen.wait(process)
print "Clump time series generation: done! [Step 2/2]"
print 'Clump time series for experiment "'+expLabel+'" DONE!'
