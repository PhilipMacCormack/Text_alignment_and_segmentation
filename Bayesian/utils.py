import cv2
import os
import matplotlib.pyplot as plt

def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def implt(img, cmp=None, t=''):
    """Show image using plt."""
    cv2.imshow('', img)
    cv2.waitKey(0)

def resize(img, scale_percent):
    """Resize image to given height."""
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

SMALL_HEIGHT = 800

def page_resize(img, height=SMALL_HEIGHT, always=False):
    """Resize image to given height."""
    if (img.shape[0] > height or always):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    
    return img

def ratio(img, height=SMALL_HEIGHT):
    """Getting scale ratio."""
    return img.shape[0] / height

def get_word_images(line_dir,line):
    directory = '{}/line{}'.format(line_dir,line)
    words = []
    for i in os.listdir(directory):
        img = cv2.imread('{}/{}'.format(directory,i),cv2.IMREAD_GRAYSCALE)
        words.append(img)
    return words

def get_gt_lines(path,file):
    with open('{}txt/{}.txt'.format(path,file),'r',encoding='utf8') as f:
        text = f.read()
        lines = text.splitlines()
        res = []
        for i,words in enumerate(lines):
            word_list = words.split()
            res.append(len(word_list))
    return res

# SECOND PREPROCESS WAY
# blur = cv2.GaussianBlur(page_image, (11,11), cv2.BORDER_DEFAULT)
# im_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
# edges = cv2.Canny(thresh, 40, 50, apertureSize=3)