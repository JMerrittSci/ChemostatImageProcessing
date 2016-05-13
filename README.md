###########################################
##               SECTIONS                ##
###########################################

1. PACKAGE REQUIREMENTS
2. DATA REQUIREMENTS
3. USAGE
4. USER VARIABLES
5. ACCREDITATION AND LICENSE

###########################################
##         PACKAGE REQUIREMENTS          ##
###########################################

- Python 2.7
- NumPy
- SciPy
- scikit-learn
- scikit-image
- (NOTE: Code has only been tested under
   Ubuntu 14.04)

###########################################
##           DATA REQUIREMENTS           ##
###########################################

Images are assumed to be 0-255 grayscale and saved
at 1288x964 in the .pgm format, with the filename
convention PREFIX-aaaaaa-bb.pgm, where "PREFIX"
is any string, "aaaaaa" is the 6-digit number of
the minute the image was taken during, and "bb"
is the 2-digit frame number of the image within
that minute. Therefore the first image in an
image set might be "PREFIX-000000-00.pgm".

NOTE: Although the above filename convention
implies up to 100 frames can be taken per minute,
the code is set up under the assumption that
there are only 5. Image sets containing more than
5 frames per minute would require modification
to many of the analysis scripts.

###########################################
##                USAGE                  ##
###########################################

After setting the user variables defined below,
running "automatic_segmentation.py" from this
directory will automatically produce time series
for cells and clumps detected in the images. Data
is saved to a folder with the experiment name
specified by the user variable "expLabel", with
cell counts in "expLabel_counts.csv" and clump
counts in "expLabel_clump_counts.csv". In both
files the first column is time in minutes (as
specified in the image filenames), the second
column are counts, and for cell counts a third
column with a standard error of counts across
frames is also included. Note that only minutes
determined to include good cell count data are
included, so some minutes may be missing from
the .csv file despite being included in the images.

In practice, the analysis scripts here do not
always correctly eliminate bad time points (such
as those dominated by large bubbles). If this is
a concern, the script "manual_segmentation_part_1.py"
should be run first. This will generate a tentative
"expLabel_counts.csv" file, which can be analyzed by
hand (or a new script) and should be saved in place
with any necessary deletions. Next, the script
"manual_segmentation_part_2.py" should be run, using
the same experiment label as in the first script.

###########################################
##           USER VARIABLES              ##
###########################################

The scripts are currently set up to be used on
the test images included. Alternatively, all
user variables should be edited by hand in
whichever of the following scripts are used:

- automatic_segmentation.py
- manual_segmentation_part_1.py
- manual_segmentation_part_2.py

The user variables are:

- expLabel: A descriptive label for the experiment.
    Determines the name of the folders and files
    generated. If left blank, defaults to the date
    the script was run.
- imgDirectory: The directory containing the images.
- filePrefix: The image filename prefix. Note that
    where filePrefix is PREFIX, the assumed filename
    convention is PREFIX-aaaaaa-bb.pgm, where "aaaaaa"
    is the 6-digit minute number the image was taken
    during and "bb" is the image's 2-digit frame number
    (note that the code currently requires there to be
    5 frames per minute). The first image of an
    experiment might be PREFIX-000000-00.pgm (the test
    images included are from later time points).
- firstMinute: Usually would be 0; note that this
    corresponds to the lowest value of "aaaaaa" of
    any image file according to the convention defined
    above.
- experimentLength: Usually the number of minutes in
    the experiment; note that this corresponds to the
    highest value of "aaaaaa" of any image file
    according to the convention defined above.
- trainingSetFilePath: The filepath of the machine
    learning training set for this strain and
    experimental condition.
- clumpBrightnessThreshold: A clump must have a max
    pixel intensity (after smoothing) of at least this
    value or be discarded. The intensity range is 0-255.
- clumpMeanBrightnessThreshold: A clump must have a
    mean pixel intensity (after smoothing) of at least
    this value or be discarded. The intensity range
    is 0-255.
- clumpSizeThreshold: A clump must have a pixel value
    area at least this high or be discarded. Note that
    the minimum clump size is already 500 pixels
    internally; clumps must have an area of at least 500
    pixels to be counted regardless of this threshold
    value.

###########################################
##       ACCREDITATION AND LICENSE       ##
###########################################

The code for the "Moments" method in the imseg.py script
was ported to Python from a port by Gabriel Landini
(for the Auto_Threshold plugin of the Fiji project for
ImageJ) of code written by M. Emre Celebi for the open
source project FOURIER:
http://fiji.sc/Auto_Threshold#Installation
https://github.com/fiji/Auto_Threshold/blob/master/src/main/java/fiji/threshold/Auto_Threshold.java
http://www.lsus.edu/faculty/~ecelebi/fourier.htm 
http://sourceforge.net/projects/fourier-ipal

Other code written by Jason Merritt and Seppe Kuehn at
the University of Illinois at Urbana-Champaign.

This code is covered by the GPLv2 license, the text of which
is included in the LICENSE file.
