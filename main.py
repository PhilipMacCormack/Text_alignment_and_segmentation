from Method_1.main_process import method_1
from Method_2.new_main_process import method_2

# REFERENCES:
    # Modified version of code structure of page hole removal & line/word segmentation based on: https://github.com/KadenMc/PreprocessingHTR
    # Code for page border removal based on: https://github.com/harshavkumar/word_segmentation
    # Bounding box auto correction based on: On-the-fly Historical Text Annotation. Vats, E., Hast, A.  In Proceedings of the 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), pp. 10-14, 2017.

# HOW TO RUN:
# 1. input the path of the directory where the image and GT (if available) is located in "PARAMETERS" below.
# 2. input the file name of the image you want to process below, do not include the file ending (only supports .jpg filetype).
# 3. If you have done bayesian optimization beforehand, set the "read_params" parameter as True.
# 4. Input optional parameters, otherwise leave them as is.

# ---------------------------------------------------------------------------------------------------------------------------------

# PARAMETERS

# Set to 1 or 2 depending on what method to be used
method = 1

# Path to data to be processed and GT
path = '../Data/Labours_Memory_Test_Data/'
# path = '../Data/iam_Test_Data/'

# Name of image to be processed (without file extension)
file = 'fac_00178_arsberattelse_1935_sid-06'
# file = 'a01-020'

# Set as True if Bayesian optimisation has been done and you want to read parameters from it, otherwise False.
read_params = False 

# Set as True if the document image has page holes in it or if unsure, else put False
holes = True

# Pre decide how you want to align the image, Transcribe='1', Linear alignment='2', IoU-based alignment='3', otherwise None
transcribe_or_gt = None

# Pre decide what value you want the slider for setting t1 to be
pre_t1 = 220

# Pre decide what value you want the slider for setting min_gap to be in method 1, if method 2 this acts as the definite value
min_gap = 27

# -------------------------------------------------------------------------------------------------------------------------------

if method == 1:
    method_1(path,file,read_params,holes,transcribe_or_gt,pre_t1,min_gap)
elif method == 2:
    method_2(path,file,read_params,holes,transcribe_or_gt,pre_t1,min_gap)