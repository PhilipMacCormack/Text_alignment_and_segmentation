import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

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
    """Resize image to given percent scale."""
    width = int(img.shape[1] * (scale_percent / 100))
    height = int(img.shape[0] * (scale_percent / 100))
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

SMALL_HEIGHT = 800

def page_resize(img, height=SMALL_HEIGHT, always=False):
    """Resize image to given height."""
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))

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

def mean_top_k(areas, k=10):
    """Get mean of the top k elements."""
    top_k_vals = min([k, len(areas)])
    return (-np.sort(-areas))[:top_k_vals].mean()

def get_iou(pred_word_bb, word_bb):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(pred_word_bb[0], word_bb[0])
    y_top = max(pred_word_bb[2], word_bb[2])
    x_right = min(pred_word_bb[1], word_bb[1])
    y_bottom = min(pred_word_bb[3], word_bb[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both AABBs
    bb1_area = (pred_word_bb[1] - pred_word_bb[0]) * (pred_word_bb[3] - pred_word_bb[2])
    bb2_area = (word_bb[1] - word_bb[0]) * (word_bb[3] - word_bb[2])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def filter1(self,min_ak,min_ad,max_height,hor_rat,max_ak,ver_rat):
        """Filters components based on height, horizontal ratio, and small area."""
        # Parameters for filter1:
        # min_ak = 20 # Top k elements with to calculate min allowed area
        # min_ad = 1.8 # divider integer to get min allowed area of component
        # max_height = (self.img.shape[0]/35)*3 # max allowed height of component
        # hor_rat = 5 # max allowed horizontal ratio of component
        # max_ak = 4 # multiplier to get max allowed area of component
        # ver_rat = 2.3 # max allowed vertical ratio of component

        # Find minimum area under which we can delete components
        self.min_area = mean_top_k(self.area, min_ak)/min_ad

        # Use bounding box area to get rid of noise/very small components
        allowed_area = np.argwhere(self.bounding_area > self.min_area)[:, 0]

        # The average paper has up to ~35 lines of text.
        # This divides the page into 35 lines, which implies that a text contour should have
        # height no more than img.height/35. To be generous, allow lines 3 times bigger than this
        allowed_height = np.argwhere(self.height <= max_height)[:, 0]
        allowed = np.intersect1d(allowed_area, allowed_height)

        # Getting rid of the remnants of horizontal lines can be done via a height to width ratio
        # If width/height is VERY high, it must be a horizontal line (and the area can't be too large)
        allowed_horizontal_ratio = np.argwhere(self.width/self.height < hor_rat)[:, 0]
        allowed = np.intersect1d(allowed, allowed_horizontal_ratio)

        allowed_max_area = np.argwhere(self.bounding_area >= self.min_area*max_ak)[:,0]
        allowed = np.intersect1d(allowed, allowed_max_area)

        allowed_vertical_ratio = np.argwhere(self.height/self.width < ver_rat)[:, 0]
        self.allowed = np.intersect1d(allowed, allowed_vertical_ratio)

        # Note: In order to draw the components from the original connected components output
        # we must track which components we're 'allowing', or keeping
        self.filter_indices(self.allowed)


# SECOND PREPROCESS WAY
# blur = cv2.GaussianBlur(page_image, (11,11), cv2.BORDER_DEFAULT)
# im_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
# edges = cv2.Canny(thresh, 40, 50, apertureSize=3)