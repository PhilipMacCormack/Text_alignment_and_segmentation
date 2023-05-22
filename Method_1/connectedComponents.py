import numpy as np
import cv2
import math
from Method_1.utils import *

def draw_border(img, border, col=(0,255,0)):
    """Draw a border on an image given the border coordinates."""
    l, r, t, b = border
    cv2.line(img, (l, t), (l, b), col, 2)
    cv2.line(img, (l, t), (r, t), col, 2)
    cv2.line(img, (l, b), (r, b), col, 2)
    cv2.line(img, (r, t), (r, b), col, 2)


def show_connected_components(img):
    """
    Displays connected components colorfully.
    Credit: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
    
    Parameters:
        img (np.ndarray): The image for which to show the connected components
    """
    ret, labels = cv2.connectedComponents(img, connectivity=8)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Convert to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img



def consecutive_groups(data, stepsize=1):
    """Finds groups of consequtive numbers."""
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def mean_top_k(areas, k=10):
    """Get mean of the top k elements."""
    top_k_vals = min([k, len(areas)])
    return (-np.sort(-areas))[:top_k_vals].mean()

def bounding_boxes_test(img,x,y,left,right,top,bottom,width,height):
    """Draws bounding box for every component."""
    # borders = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    # borders = img
    for i in range(len(x)):
        draw_border(img, (left[i], right[i], top[i], bottom[i]), col=255)

    bounding_rect = np.stack([x, y, width, height], axis=1)
    return img

def filter_indices_test(allowed,left,right,top,bottom,area,x,y,width,height,bounding_area):
    """Filters statistics by indices in allowed."""
    left = left[allowed]
    right = right[allowed]
    top = top[allowed]
    bottom = bottom[allowed]
    area = area[allowed]
    x = x[allowed]
    y = y[allowed]
    width = width[allowed]
    height = height[allowed]
    bounding_area = bounding_area[allowed]
    return left,right,top,bottom,area,x,y,width,height,bounding_area

def filter1_test(min_ak,min_ad,max_height,hor_rat,max_ak,ver_rat,area,bounding_area,height,width):
        """Filters components based on height, horizontal ratio, and small area."""
        # Parameters for filter1:
        # min_ak = 20 # Top k elements with to calculate min allowed area
        # min_ad = 1.8 # divider integer to get min allowed area of component
        # max_height = (self.img.shape[0]/35)*3 # max allowed height of component
        # hor_rat = 5 # max allowed horizontal ratio of component
        # max_ak = 4 # multiplier to get max allowed area of component
        # ver_rat = 2.3 # max allowed vertical ratio of component

        # Find minimum area under which we can delete components
        min_area = mean_top_k(area, min_ak)/min_ad

        # Use bounding box area to get rid of noise/very small components
        allowed_area = np.argwhere(bounding_area > min_area)[:, 0]

        # The average paper has up to ~35 lines of text.
        # This divides the page into 35 lines, which implies that a text contour should have
        # height no more than img.height/35. To be generous, allow lines 3 times bigger than this
        allowed_height = np.argwhere(height <= max_height)[:, 0]
        allowed = np.intersect1d(allowed_area, allowed_height)

        # Getting rid of the remnants of horizontal lines can be done via a height to width ratio
        # If width/height is VERY high, it must be a horizontal line (and the area can't be too large)
        allowed_horizontal_ratio = np.argwhere(width/height < hor_rat)[:, 0]
        allowed = np.intersect1d(allowed, allowed_horizontal_ratio)

        allowed_max_area = np.argwhere(bounding_area >= min_area*max_ak)[:,0]
        allowed = np.intersect1d(allowed, allowed_max_area)

        allowed_vertical_ratio = np.argwhere(height/width < ver_rat)[:, 0]
        allowed = np.intersect1d(allowed, allowed_vertical_ratio)

        # Note: In order to draw the components from the original connected components output
        # we must track which components we're 'allowing', or keeping
        return allowed

class Components():
    """
    Class to organize connected components and related statistics.

    Methods
    -------
    filter1():
        Filters components based on height, horizontal ratio, and small area.

    filter_strays(y):
        Filters 'stray' components by according to y-value/closeness to other components.

    filter2():
        Filters components based on closeness to other components (stray) and smaller area.

    filter(config):
        Filters components.
    """

    def __init__(self, img):
        self.img = img
        self.nb_components, self.output, self.stats, self.centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        self.nb_components -= 1

        self.left = self.stats[1:, 0]
        self.top = self.stats[1:, 1]
        self.width = self.stats[1:, 2]
        self.height = self.stats[1:, 3]
        self.area = self.stats[1:, 4]
        self.bounding_area = self.width * self.height
        self.right = self.left + self.width
        self.bottom = self.top + self.height

        self.x = self.left + self.width//2
        self.y = self.top + self.height//2

        self.diagonal = math.sqrt(
            math.pow(self.output.shape[0], 2) + math.pow(self.output.shape[1], 2))

    def __len__(self):
        return len(self.x)

    def bounding_boxes(self):
        """Draws bounding box for every component."""
        borders = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        for i in range(len(self.x)):
            draw_border(borders, (self.left[i], self.right[i], self.top[i], self.bottom[i]), col=255)

        self.bounding_rect = np.stack([self.x, self.y, self.width, self.height], axis=1)
        return borders

    def filter_indices(self, allowed):
        """Filters statistics by indices in allowed."""
        self.left = self.left[allowed]
        self.right = self.right[allowed]
        self.top = self.top[allowed]
        self.bottom = self.bottom[allowed]
        self.area = self.area[allowed]
        self.x = self.x[allowed]
        self.y = self.y[allowed]
        self.width = self.width[allowed]
        self.height = self.height[allowed]
        self.bounding_area = self.bounding_area[allowed]

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


    def filter_strays(self, y):
        """Filters 'stray' components by according to y-value/closeness to other components."""
        counts, boundaries = np.histogram(y, bins=40)

        # Find runs of non-zero counts
        non_zero = np.argwhere(counts != 0)[:, 0]
        consecutive = consecutive_groups(non_zero)

        check_boundaries = []
        indices = []
        for c in consecutive:
            # print('c: ',c)
            # Check consecutive interval length
            if len(c) <= 10:
                # Check number of components in consequtive interval
                if counts[c].sum() <= 4:
                    for b in c:
                        indices.extend(np.argwhere(np.logical_and(y >= boundaries[b], y <= boundaries[b + 1]))[:, 0])
        
        return np.array(indices)


    def filter2(self):
        """Filters components based on closeness to other components (stray) and smaller area."""
        # Get components with 'small enough' area - (average of top k areas)/1.5
        small_area_indices = np.argwhere(self.area <= mean_top_k(self.area, k=15)/1.5)

        # Get stray components
        stray_indices = self.filter_strays(self.y)

        # Combine filtering - If small area and a stray, then get rid of it!
        remove = np.intersect1d(small_area_indices, stray_indices)

        # Get 'allowed' indices - compliment of removed
        allowed = np.setdiff1d(np.array([i for i in range(len(self.area))]), remove)
        self.allowed = self.allowed[allowed]
        
        # Note: In order to draw the components from the original connected components output
        # we must track which components we're 'allowing', or keeping
        self.filter_indices(allowed)

    def filter(self,file):
        """Filters components."""
        self.borders = self.bounding_boxes()
        # cv2.imwrite('results/{}/components_borders0.png'.format(file), self.borders)
        # Filters based on height, horizontal ratio, vertical ratio, and very small area

        # def nothing(x):
        #     pass
        # cv2.namedWindow('filter1')
        # cv2.createTrackbar('hor_rat','filter1',150,200,nothing)
        # cv2.createTrackbar('ver_rat','filter1',27,150,nothing)
        # cv2.createTrackbar('min_ak','filter1',69,100,nothing)
        # cv2.createTrackbar('max_ak','filter1',2,15,nothing)
        # cv2.createTrackbar('min_ad','filter1',22,70,nothing)
        # cv2.createTrackbar('max_height','filter1',158,250,nothing)
        # cv2.setTrackbarMin('min_ak','filter1', 1)
        # cv2.setTrackbarMin('min_ad','filter1', 3)
        # cv2.setTrackbarMin('max_height','filter1', 55)
        # cv2.setTrackbarMin('hor_rat','filter1', 10)
        # cv2.setTrackbarMin('max_ak','filter1', 1)
        # cv2.setTrackbarMin('ver_rat','filter1', 5)

        # while(1):
        #     aarea = self.area
        #     bbounding_area = self.bounding_area
        #     hheight = self.height
        #     wwidth = self.width
        #     lleft = self.left
        #     rright = self.right
        #     ttop = self.top
        #     bbottom = self.bottom
        #     xx = self.x
        #     yy = self.y
        #     imgg = self.img
        #     no_noise_img = cv2.imread("results/{}/no_noise_img.png".format(file))
        #     allowedd = filter1_test(cv2.getTrackbarPos('min_ak','filter1'),cv2.getTrackbarPos('min_ad','filter1')/10,cv2.getTrackbarPos('max_height','filter1'),cv2.getTrackbarPos('hor_rat','filter1')/10,cv2.getTrackbarPos('max_ak','filter1'),cv2.getTrackbarPos('ver_rat','filter1')/10,aarea,bbounding_area,hheight,wwidth)
        #     llleft,rrright,tttop,bbbottom,aaarea,xxx,yyy,wwwidth,hhheight,bbbounding_area = filter_indices_test(allowedd,lleft,rright,ttop,bbottom,aarea,xx,yy,wwidth,hheight,bbounding_area)
        #     bborders = bounding_boxes_test(no_noise_img,xxx,yyy,llleft,rrright,tttop,bbbottom,wwwidth,hhheight)
        #     resized =resize(bborders, 15)
        #     cv2.imshow('filter_1',resized)
        #     k = cv2.waitKey(1) & 0xFF
        #     if k == 27:
        #         break 
        # min_ak = cv2.getTrackbarPos('min_ak','filter1')
        # min_ad = cv2.getTrackbarPos('min_ad','filter1')/10
        # max_height = cv2.getTrackbarPos('max_height','filter1')
        # hor_rat = cv2.getTrackbarPos('hor_rat','filter1')/10
        # max_ak = cv2.getTrackbarPos('max_ak','filter1')
        # ver_rat = cv2.getTrackbarPos('ver_rat','filter1')/10
        # print('hor_rat: ',hor_rat)
        # print('ver_rat: ',ver_rat)
        # print('min_ak: ',min_ak)
        # print('max_ak: ',max_ak)
        # print('min_ad: ',min_ad)
        # print('max_height: ',max_height)
        hor_rat = 150/10
        ver_rat = 27/10
        min_ak = 69
        max_ak = 2
        min_ad = 22/10
        max_height = 158

        self.filter1(min_ak,min_ad,max_height,hor_rat,max_ak,ver_rat)

        # Draw connected components after filter1
        self.filtered = np.zeros((self.img.shape))
        for i in range(len(self.allowed)):
            self.filtered[self.output == self.allowed[i] + 1] = 255

        # Draw bounding boxes after filter1
        self.borders = self.bounding_boxes()

        # # Save intermediate images
        # config['save_inter_func'](config, self.filtered, "components_filtered1")
        # config['save_inter_func'](config, self.borders, "components_borders1")
        # cv2.imwrite('results/{}/components_filtered1.png'.format(file), self.filtered)
        # cv2.imwrite('results/{}/components_borders1.png'.format(file), self.borders)


        # Filters based on closeness to other components (stray) and smaller area
        # self.filter2()
        
        # Draw connected components after filter2
        self.filtered = np.zeros((self.img.shape))
        for i in range(len(self.allowed)):
            self.filtered[self.output == self.allowed[i] + 1] = 255

        # Draw bounding boxes after filter2
        self.borders = self.bounding_boxes()

        # config['save_inter_func'](config, self.filtered, "components_filtered2")
        # config['save_inter_func'](config, self.borders, "components_borders2")
        self.filtered_comp_img = self.filtered
        cv2.imwrite('results/{}/components_filtered2.png'.format(file), self.filtered)
        # cv2.imwrite('results/{}/components_borders2.png'.format(file), self.borders)


# Creates connected components
def connected_components(img,file):
    """Create, visualize, filter, and return connected components."""
    # Save a connected components display image
    components_labeled_img = show_connected_components(img)
    # config['save_inter_func'](config, components_labeled_img, "components_labeled")
    # cv2.imwrite('results/{}/components_labeled_img.png'.format(file), components_labeled_img)
    
    # Create, filter, and return connected components
    components = Components(img)
    components.filter(file)
    return components