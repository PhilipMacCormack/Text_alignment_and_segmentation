import anvil.server
from PIL import Image
import io
import cv2
import numpy as np
from page_segment import page_segment
from utils import resize
import math
from screeninfo import get_monitors
from utils import page_resize
from hole_remove import page_hole_removal
from scipy.signal import find_peaks

anvil.server.connect('server_4PRKH5EHSULUI43XTN55LW3Y-RZP62GNYGVWSV74L')

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

    def filter(self):
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


# Creates connected components
def connected_components(img):
    """Create, visualize, filter, and return connected components."""
    # Save a connected components display image
    components_labeled_img = show_connected_components(img)
    # config['save_inter_func'](config, components_labeled_img, "components_labeled")
    # cv2.imwrite('results/{}/components_labeled_img.png'.format(file), components_labeled_img)
    
    # Create, filter, and return connected components
    components = Components(img)
    components.filter()
    return components

def getHorizontalProjectionProfileSmoothed(image):
    horizontal_projection = np.sum(image==255, axis = 1) 
    box = np.ones(40)/40
    smooth = np.convolve(horizontal_projection,box,mode='valid')
    return smooth

def roll_zero_pad(a, shift, axis=None):
    """Shifts an array left if shift < 0 and right if shift > 0, padding the new elements with 0."""
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def get_gap_from_sums(sums, thresh):
    """Returns interval gaps in the row or column sums of an image. Ignores 'edge' gaps.

    Parameters:
    sums (np.ndarray): An array (dtype == np.bool) representing whether there is any pixels in a column.
    """
    data = np.argwhere(sums != 0)[:, 0]
    consecutive = np.split(data, np.where(np.diff(data) != 1)[0] + 1)

    if len(consecutive[0]) == 0:
        return []

    # Characterize consequtive runs of white columns as gaps
    # Get the start and end of each gap, but ignore any gaps on the edge of an image
    return [[c[0], c[-1]] for c in consecutive if c[0] != 0 and c[-1] != thresh.shape[1]-1]

def get_sums(img):
    """Determine whether there are any black pixels in each column of an image.

    Parameters:
    img (np.ndarray): An 2D image where each pixel has value 0 or 255.
    """
    return np.invert(((255-img).sum(axis=0)).astype(np.bool_))

def get_gaps(img, degree_slant=30):
    """Gets gaps in an image, a slanted verion of the image, and the intersection of those gaps."""
    height = img.shape[0]
    width = img.shape[1]

    # Get gaps in image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sums = get_sums(img.astype(np.uint8))
    gaps = get_gap_from_sums(sums, img)

    #cv2.imshow("img_line", np.tile(sums.astype(np.uint8)*255, (100, 1)))

    # Get gaps in slanted image - image is slanted theta degrees
    # This is very necessary for people who write with a forward slant
    theta = np.radians(degree_slant)
    rolled = img.copy()
    for i in range(height):
        hyp = int(np.round(i*np.tan(theta)))
        rolled[i, :] = roll_zero_pad(rolled[i, :], hyp)

    '''cv2.imshow('rolled', rolled)'''

    sums_slanted = get_sums(rolled)

    #cv2.imshow("sums_slanted", np.tile(sums_slanted.astype(np.uint8)*255, (100, 1)))

    # Shift slanted image - Slanted 45 degree line '/'. Calculate the subtraction to
    # bring the gaps back in line with those original image
    subtract = int((height * np.cos((np.pi/2) - theta))//2)
    sums_slanted = roll_zero_pad(sums_slanted, -subtract)
    gaps_slanted = get_gap_from_sums(sums_slanted, rolled)

    #cv2.imshow("sums_slanted-subtracted", np.tile(sums_slanted.astype(np.uint8)*255, (100, 1)))

    # Get intersection of gaps in the image and the slanted image
    sums_both = (np.logical_and(sums_slanted.astype(np.bool_), sums.astype(np.bool_)))
    gaps_both = get_gap_from_sums(sums_both, img)

    #cv2.imshow("both_line", np.tile(sums_both.astype(np.uint8)*255, (100, 1)))

    return gaps, gaps_slanted, gaps_both

def find_nearest_value_index(array, value):
    """Finds the nearest value to those in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_middle_delete_unmatched(middles1, middles2, min_gap):
    """Takes two arrays of space dividers and removes dividers from middles1 which aren't in middles2."""
    if len(middles1) == 0 or len(middles2) == 0:
        return []

    middles1 = np.array(middles1)
    middles2 = np.array(middles2)
    nearest = np.array([find_nearest_value_index(middles2, m) for m in middles1])
    diff = np.abs(middles2[nearest] - middles1)
    middles_final = middles1[np.argwhere(diff < min_gap)[:, 0]]
    return middles_final

def filter_middles(gaps, min_gap):
    """Filters gaps smaller than some minimum gap threshold."""
    middles = [(g[0] + g[1])//2 for g in gaps]
    ranges = [g[1] - g[0] for g in gaps]
    return [m for i, m in enumerate(middles) if ranges[i] > min_gap]

def get_middle(img, gaps, gaps_slanted, gaps_both, min_gap):
    """Calculates reasonable space dividers in a line (middles) provided its gaps."""
    # Get middles
    middles = filter_middles(gaps, min_gap)
    middles_slanted = filter_middles(gaps_slanted, min_gap*1.13)
    middles_both = filter_middles(gaps_both, min_gap*0.88)

    # Merge the multiple analyses into one
    middles.extend(middles_slanted)
    middles.extend(middles_both)
    middles_merged = np.array(middles)
    middles_merged.sort()

    if len(middles_merged) == 0:
        return [], img

    merge_sum = middles_merged[0]
    merge_count = 1
    middles_final = []
    for i in range(1, len(middles_merged)):
        if middles_merged[i] - middles_merged[i - 1] < min_gap:
            # print('middlesi - middlesi+1: ', middles_merged[i] - middles_merged[i - 1])
            merge_sum += middles_merged[i]
            merge_count += 1
        else:
            middles_final.append(int(np.round(merge_sum/merge_count)))
            merge_sum = middles_merged[i]
            merge_count = 1
    
    middles_final.append(int(np.round(merge_sum/merge_count)))  
    # print('middles_final: ',middles_final)
    return middles_final, img

def remove_vertical_components(components):
    """Removes vertically skinny components, which are often unwanted lines/artifacts in the image."""
    w = components.right - components.left
    h = components.bottom - components.top
    h_w_rat = 1.7
    # Return components with an acceptable h/w ratio
    return [np.argwhere(h/w < h_w_rat)[:, 0]]

def line_segmentation(no_noise, components, scale_percent,page_image):
    horizontal = getHorizontalProjectionProfileSmoothed(components.filtered)
    peaks, _ = find_peaks(horizontal,height=16,distance=100)
    # plt.plot(horizontal)
    # plt.plot(peaks, horizontal[peaks], 'x')
    # plt.show()
    # print(peaks)
    i = len(peaks)
    print('lines found: ',i)
    kernel_size = 1000
    kernel = np.zeros((kernel_size,kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    testimg = cv2.cvtColor(no_noise.copy(),cv2.COLOR_GRAY2RGB)
    line_components = [[] for i in range(len(peaks))]
    for i,cy1 in enumerate(components.top):
        cy2 = components.bottom[i]
        cx1 = components.left[i]
        cx2 = components.right[i]
        coord = [cx1,cx2,cy1,cy2]
        y_diffs = []
        cv2.rectangle(testimg, (components.left[i], cy1), (components.right[i],cy2), (0,255,0),2)
        for j,y in enumerate(peaks):
            cv2.line(testimg,(0,y),(page_image.shape[1],y),(0,0,255),2)
            y_diff = abs(((cy2+cy1)/2)-y)
            y_diffs.append(y_diff)
        best_line_for_comp = y_diffs.index(min(y_diffs))
        line_components[best_line_for_comp].append(coord)
    # plt.imshow(testimg, cmap='gray')
    # plt.show()
    # cv2.imwrite('results/{}/line_y.png'.format(file), testimg)

    line_seg = page_image.copy()
    line_coords = []
    for line_comps in line_components:
        x1 = min(x[0] for x in line_comps)
        x2 = max(x[1] for x in line_comps)
        y1 = min(y[2] for y in line_comps)
        y2 = max(y[3] for y in line_comps)
        line_coords.append([x1,x2,y1,y2])

    line_seg = page_image.copy()
    for coord in line_coords:
        cv2.rectangle(line_seg, (coord[0],coord[2]),(coord[1],coord[3]),(0,255,0),2)

    def inside(bl, tr, p) :
        if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
            return True
        else :
            return False

    remove_line_boxes = []
    resized_lineimg = resize(line_seg, scale_percent)
    
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            remove_line_boxes.append([x/(scale_percent/100),y/(scale_percent/100)])
            cv2.circle(resized_lineimg, (x,y),4,(0,0,255), -20)
            cv2.imshow('remove_win', resized_lineimg)

    print("Left click the line boxes you want to correct to mark them. Press 'Q' when you are satisfied.")
    while(1):
        cv2.imshow('remove_win', resized_lineimg)
        cv2.setMouseCallback('remove_win', click_event)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    if remove_line_boxes != []:
        for point in remove_line_boxes:
            inside_areas = []
            inside_inds = []
            for ind, prev_box in enumerate(line_coords):
                if inside((prev_box[0],prev_box[2]),(prev_box[1],prev_box[3]),(point[0],point[1])):
                    inside_areas.append((prev_box[2]-prev_box[0])*(prev_box[3]-prev_box[2]))
                    inside_inds.append(ind)
            if inside_inds != []:
                if len(inside_areas) > 1:
                    selected_inside_val = inside_areas.index(min(inside_areas))
                    selected_inside_ind = inside_inds[selected_inside_val]
                    del line_coords[selected_inside_ind]
                    # del lines[selected_inside_ind]
                else:
                    del line_coords[inside_inds[0]]
                    # del[lines[inside_inds[0]]]

    new_line_image = page_image.copy()
    for coord in line_coords:
        cv2.rectangle(new_line_image,(coord[0],coord[2]),(coord[1],coord[3]), (0,255,0),4)

    resized2 = resize(new_line_image, scale_percent)
    import selectinwindow
    import sys
    sys.setrecursionlimit(10 ** 9)
    wName = 'Add line bounding-boxes'
    rectI = selectinwindow.DragRectangle(resized2, wName, new_line_image.shape[0], new_line_image.shape[1])
    cv2.namedWindow(rectI.wname)
    cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)
    print("Add new line box by clicking & dragging, press & hold 'Enter' when you are satisfied with a box. Press 'Q' when you are finished.")
    cache_images = []
    cache_line_images = []
    while(1):
        # display the image
        cv2.imshow(wName, rectI.image)
        key = cv2.waitKey(1)
        if key == 13:
            x1 = int(rectI.outRect.x/(scale_percent/100))
            x2 = int(rectI.outRect.x/(scale_percent/100)+rectI.outRect.w/(scale_percent/100))
            y1 = int(rectI.outRect.y/(scale_percent/100))
            y2 = int(rectI.outRect.y/(scale_percent/100)+rectI.outRect.h/(scale_percent/100))
            if (y1+y2)/2 == 0.0:
                pass
            else:
                y_mean = (y1+y2)/2
                new_box = [x1,x2,y1,y2]
                if new_box not in line_coords: # Prevents multiple entries of same box
                    cache_images.append(rectI.image.copy())
                    cache_line_images.append(new_line_image.copy())
                    cv2.rectangle(new_line_image,(x1,y1),(x2,y2), (0,255,0),4)
                    cv2.rectangle(rectI.image, (int(x1*(scale_percent/100)), int(y1*(scale_percent/100))), (int(x2*(scale_percent/100)), int(y2*(scale_percent/100))), (0,255,0),1)
                    line_coords.append(new_box)
                    # line = Line_added(components, x1,x2,y1,y2, no_noise, file)
                    # lines.append(line)
                    print('Line box registered.')
                rectI.reset()
        if key == 8:
            if cache_images != []:
                rectI.image = cache_images[-1]
                new_line_image = cache_line_images[-1]
                del cache_images[-1]
                del line_coords[-1]
                del cache_line_images[-1]

        if key == ord('q'):
            break

    # cv2.imwrite('results/{}/corrected_line_image.png'.format(file), cv2.cvtColor(new_line_image,cv2.COLOR_BGR2RGB))

    cv2.destroyAllWindows()
    line_coords.sort(key = lambda x: x[3])
    return line_coords

def word_segmentation(line_coords, no_noise, min_gap, scale_percent):
    word_coords = []
    for i,coord in enumerate(line_coords):
        line_no = i+1
        line_img = no_noise[coord[2]:coord[3], coord[0]:coord[1]]
        line_img_temp = no_noise[coord[2]-10:coord[3]+10, coord[0]-10:coord[1]+10]
        # print('line :', i+1)

        line_img_edges = cv2.Canny(line_img, 255, 255/3)
        components = connected_components(line_img_edges)
        # filt_components = remove_vertical_components(components)
        # comp_box_line = line_img.copy()
        # for i,left in enumerate(components.left):
        #     cv2.rectangle(comp_box_line, (left,components.top[i]),(components.right[i], components.bottom[i]),(0,0,0),-1)
        # plt.imshow(comp_box_line, cmap='gray')
        # plt.show()

        # row, col = line_img_temp.shape[:2]
        # bottom = line_img_temp[row - 2:row, 0:col]
        # bordersize = 1
        # line_img_temp = cv2.copyMakeBorder(line_img_temp,top=bordersize,bottom=bordersize,left=bordersize,right=bordersize,borderType=cv2.BORDER_CONSTANT,value=0)
        # cv2.floodFill(line_img_temp, None, (0,0), 255)
        # y, x = line_img_temp.shape
        # line_img_temp = line_img_temp[1:y-1, 1:x-1]
        # y, x = line_img_temp.shape
        # line_img_temp = line_img_temp[10:y-10, 10:x-10]
        # plt.imshow(line_img_temp, cmap='gray')
        # plt.show()


        comp_img = components.filtered_comp_img
        comp_img = cv2.bitwise_not(comp_img.astype(np.uint8))
        # cv2.imshow('testcomp', resize(comp_img,25))
        # cv2.waitKey(0)
        gaps,gaps_slanted,gaps_both = get_gaps(comp_img)
        # print(i+1,gaps_slanted)

        word_test = cv2.cvtColor(line_img.copy(),cv2.COLOR_GRAY2BGR)
        middles, thresh = get_middle(line_img, gaps, gaps_slanted, gaps_both, min_gap)
        test_comp = cv2.cvtColor(line_img.copy(),cv2.COLOR_GRAY2BGR)
        for i,left in enumerate(components.left):
            cv2.rectangle(test_comp, (left, components.top[i]), (components.right[i], components.bottom[i]), (0,255,0), 2)
            cv2.line(test_comp, (components.x[i],components.top[i]),(components.x[i],components.bottom[i]),(0,0,255),2)
        # for mid in middles:
        #     cv2.rectangle(test_comp, (mid, 0),(mid, coord[3]),(255,0,0),2)
        # plt.imshow(test_comp, cmap='gray')
        # plt.show()
        if len(middles) == 0:
            cv2.rectangle(word_test, (coord[0],coord[2]), (coord[1], coord[3]),(0,255,0), 2)
            word_coords.append([line_no,[coord[0],coord[1],coord[2],coord[3]]])
            # plt.imshow(word_test, cmap='gray')
            # plt.show()
        else:
            # [filt_components] = filt_components
            x_components = components.x
            middles = np.append(middles, coord[1]-coord[0])
            middles.sort()
            segments_x = []
            for i,mid in enumerate(middles):
                if mid == middles[0]:
                    segments_x.append([0,mid])
                else:
                    segments_x.append([middles[i-1], mid])

            # print('filt_comps: ', filt_components)
            # print('len xline: ', len(x_line))
            # print('middles: ', middles)
            # print('segments_x: ', segments_x)
            # print('x_line: ', x_components)
            segments = [[] for i in range(len(middles))]
            # print('segments: ', segments)


            for i,seg_x in enumerate(segments_x):
                for j,x_comp in enumerate(x_components):
                    if x_comp > seg_x[0] and x_comp < seg_x[1]:
                        segments[i].append(j)

            # print('segments after: ', segments)
            for s in segments:
                x1s = components.left[s]
                x2s = components.right[s]
                y1s = components.top[s]
                y2s = components.bottom[s]
                if len(x1s) > 0:
                    x1 = x1s.min()
                    x2 = x2s.max()
                    y1 = y1s.min()
                    y2 = y2s.max()
                    # print('(x1,x2): ', x1, x2)
                    cv2.rectangle(word_test, (x1,y1), (x2,y2), (0,255,0), 2)
                    word_pos = [x1+coord[0],x2+coord[0],y1+coord[2],y2+coord[2]]
                    word_coords.append([line_no, word_pos])
            # plt.imshow(word_test, cmap='gray')
            # plt.show()
    word_seg = cv2.cvtColor(no_noise.copy(), cv2.COLOR_GRAY2RGB)
    # plt.imshow(word_seg,cmap='gray')
    # plt.show()
    return word_coords

def nothing(x):
	pass

def noise_removal(image,thresh_param,kernel_size):
    thresh, image = cv2.threshold(image, thresh_param, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

@anvil.server.callable
def segment_image(image):
    image_object = Image.open(io.BytesIO(image.get_bytes()))
    image = np.array(image_object)
    holes = True
    min_gap = 25

    monitors = []
    for m in get_monitors():
        if len(get_monitors()) == 1:
            height = m.height
        else:
            monitors.append(m.height)
    if monitors != []:
        height = max(monitors) - 50

    # Remove borders from document
    page_image,s_points,t_points,M = page_segment(image)

    # get scale percent to fit image to monitor size
    orig_height = page_image.shape[0]
    height_check_img = page_resize(page_image, height)
    new_height = height_check_img.shape[0]
    scale_percent = (new_height/orig_height)*100
    print('image scaled to screen height, scale percent: ', scale_percent)

    # Remove page holes if any
    if holes:
        gray = page_hole_removal(page_image)
    else:
        gray = cv2.cvtColor(page_image,cv2.COLOR_BGR2GRAY)

    # remove noise
    cv2.namedWindow('noise_remove')
    cv2.createTrackbar('t1','noise_remove',212,235,nothing)
    cv2.setTrackbarMin('t1','noise_remove', 50)
    print("Choose a suitable threshold for the document. Press 'Q' when you are satisfied.")
    while(1):
        no_noise = noise_removal(gray,cv2.getTrackbarPos('t1','noise_remove'),1)
        resized = resize(no_noise, scale_percent)
        cv2.imshow('noiseremove',resized)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break 
    t1 = cv2.getTrackbarPos('t1','noise_remove')
    print('Threshold set to: ',t1)
    cv2.destroyAllWindows()

    no_noise = noise_removal(gray,t1, 1)

    # remove noise at borders of image
    row, col = no_noise.shape[:2]
    bottom = no_noise[row - 2:row, 0:col]
    bordersize = 1
    no_noise = cv2.copyMakeBorder(no_noise,top=bordersize,bottom=bordersize,left=bordersize,right=bordersize,borderType=cv2.BORDER_CONSTANT,value=0)
    cv2.floodFill(no_noise, None, (0,0), 255)
    y, x = no_noise.shape
    no_noise = no_noise[1:y-1, 1:x-1]

    edges = cv2.Canny(no_noise, 255, 255/3)
    
    #Segment image into lines & words
    components = connected_components(edges)

    # line_coords = GMM_line_segmentation(no_noise, components, scale_percent,page_image, file, path)  # GMM LINE SEGMENTATION METHOD

    line_coords = line_segmentation(no_noise, components, scale_percent, page_image) # MY LINE SEGMENTATION METHOD
    word_coords = word_segmentation(line_coords, no_noise, scale_percent, min_gap) # MY WORD SEGMENTATION

    words_image = cv2.cvtColor(no_noise.copy(),cv2.COLOR_GRAY2BGR)
    print('word coords: ', word_coords)
    for coord in word_coords:
        cv2.rectangle(words_image, (coord[1][0], coord[1][2]), (coord[1][1], coord[1][3]), (0,255,0), 2)

    return_image = Image.fromarray(words_image.astype('uint8'))
    bs = io.BytesIO()
    return_image.save(bs, format="JPEG")
    return anvil.BlobMedia("image/jpg", bs.getvalue(), name='page_image')


if __name__ == "__main__":
    try:
        anvil.server.wait_forever()
    except KeyboardInterrupt:
        print('... Connection to Anvil uplink closed.')
        