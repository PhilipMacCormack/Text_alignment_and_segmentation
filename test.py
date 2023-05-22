import os

# Path to data to be processed and GT
# path = '../Data/Labours_Memory_Test_Data/'
path = '../Data/iam_Test_Data/'

# Name of image to be processed (without file extension)
# file = 'fac_00178_arsberattelse_1935_sid-06'
file = 'a01-020'

if os.path.isfile('{}{}.jpg'.format(path,file)):
    extension = '.jpg'

elif os.path.isfile('{}{}.png'.format(path,file)):
    extension = '.png'

print(extension)
