import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from utils import *
import math
from utils import page_resize, ratio

def get_word_vector(path,file):
    with open('{}txt/{}.txt'.format(path,file),'r',encoding='utf8') as f:
        text = f.read()
        utf8_vector = text.split()
    return utf8_vector

def page_segment(image):
    plt.rcParams['figure.figsize'] = (9.0, 9.0)

    # Loading image and converting to RGB from BGR
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def edges_det(img, min_val, max_val):
        """ Preprocessing (gray, thresh, filter, border) + Canny edge detection """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applying blur and threshold
        img = cv2.bilateralFilter(img, 9, 75, 75)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

        # Median blur replace center pixel by median of pixels under kelner
        # => removes thin details
        img = cv2.medianBlur(img, 11)

        # Add black border - detection of border touching pages
        # Contour can't touch side of image
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return cv2.Canny(img, min_val, max_val)

    small = page_resize(img)
    # Edge detection ()
    edges_image = edges_det(small, 200, 250)

    # Close gaps between edges (double page clouse => rectangle kernel)
    edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))

    def four_corners_sort(pts):
        """ Sort corners: top-left, bot-left, bot-right, top-right"""
        diff = np.diff(pts, axis=1)
        summ = pts.sum(axis=1)
        return np.array([pts[np.argmin(summ)],
                        pts[np.argmax(diff)],
                        pts[np.argmax(summ)],
                        pts[np.argmin(diff)]])


    def contour_offset(cnt, offset):
        """ Offset contour because of 5px border """
        cnt += offset
        cnt[cnt < 0] = 0
        return cnt


    def find_page_contours(edges, img):
        """ Finding corner points of page contour """
        # Getting contours  
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Finding biggest rectangle otherwise return original corners
        height = edges.shape[0]
        width = edges.shape[1]
        MIN_COUNTOUR_AREA = height * width * 0.5
        MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

        max_area = MIN_COUNTOUR_AREA
        page_contour = np.array([[0, 0],
                                [0, height-5],
                                [width-5, height-5],
                                [width-5, 0]])

        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

            # Page has 4 corners and it is convex
            if (len(approx) == 4 and
                    cv2.isContourConvex(approx) and
                    max_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
                
                max_area = cv2.contourArea(approx)
                page_contour = approx[:, 0]

        # Sort corners and offset them
        page_contour = four_corners_sort(page_contour)
        return contour_offset(page_contour, (-5, -5))

    page_contour = find_page_contours(edges_image, small)

        
    # Recalculate to original scale
    page_contour = page_contour.dot(ratio(img, small.shape[0]))

    def persp_transform(img, s_points):
        """ Transform perspective from start points to target points """
        # Euclidean distance - calculate maximum height and width
        height = max(np.linalg.norm(s_points[0] - s_points[1]),
                    np.linalg.norm(s_points[2] - s_points[3]))
        width = max(np.linalg.norm(s_points[1] - s_points[2]),
                    np.linalg.norm(s_points[3] - s_points[0]))
        
        # Create target points
        t_points = np.array([[0, 0],
                            [0, height],
                            [width, height],
                            [width, 0]], np.float32)
        
        # getPerspectiveTransform() needs float32
        if s_points.dtype != np.float32:
            s_points = s_points.astype(np.float32)
        
        M = cv2.getPerspectiveTransform(s_points, t_points) 
        return cv2.warpPerspective(img, M, (int(width), int(height))), s_points, t_points
        
    #s_points: top_left, bot_left, bot_right, top_right
    newImage, s_points, t_points = persp_transform(img, page_contour)

    return [newImage,s_points, t_points]

def get_equidistant_circle_points(r, num_points=8):
    """Gets equidistant points on a circle."""
    points = []
    for index in range(num_points):
        points.append([r*math.cos((index*2*math.pi)/num_points),
                      r*math.sin((index*2*math.pi)/num_points)])
    return points


def page_hole_removal(img):
    """Removes page ring holes, should they exist.
    
    Parameters:
        img (np.ndarray): The image for which to remove page holes.
    """

    img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = cv2.SimpleBlobDetector_create()
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = (img.shape[0]*img.shape[1])*0.0000005
    # params.minArea = 5
    params.maxArea = (img.shape[0]*img.shape[1])*0.005
    # params.maxArea = 90000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.05

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    # Distance Between Blobs
    params.minDistBetweenBlobs = img.shape[0]//50

    # Color of blobs
    # params.filterByColor = True

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    keypoints = detector.detect(img)

    # Number of points around circle to sample color
    num_points = 20

    # Used when we try to sample a color off-screen
    adopt_next = False

    # Iterate over the circles
    for k in keypoints:

        # Calculate the average color around the circle
        color_points = get_equidistant_circle_points(
            (k.size/2)*1.15, num_points=num_points)
        colors = np.zeros((num_points, 3))
        for c, p in enumerate(color_points):
            try:
                colors[c] = img[int(
                    k.pt[1]) + int(p[1]), int(k.pt[0]) + int(p[0])]

                if adopt_next:
                    i = 1
                    while np.all((colors[c - i] == 0)):
                        colors[c - i] = colors[c]
                        i -= 1
                    adopt_next = False

            except:
                # Adopt previous (if the previous isn't also nothing)
                if c > 0 and not np.all((colors[c - 1] == 0)):
                    colors[c] = colors[c - 1]
                # Signal to adopt the next color
                else:
                    adopt_next = True
        
        color = colors.mean(axis=0).astype(int)
        color = (int(color[0]), int(color[1]), int(color[2]))

        # Fill the circle with the average color around it
        cv2.circle(img, (int(k.pt[0]), int(
            k.pt[1])), int((k.size/2)*1.15), color, -1)

        # Blur the area around the circle to avoid it being picked up by canny
        left = int(k.pt[0]) - int((k.size/2)*1.2)
        right = int(k.pt[0]) + int((k.size/2)*1.2)
        top = int(k.pt[1]) - int((k.size/2)*1.2)
        bottom = int(k.pt[1]) + int((k.size/2)*1.2)
        img_to_blur = img[top:bottom, left:right]
        blurred = cv2.GaussianBlur(img_to_blur, (15, 15), 0)
        img[top:bottom, left:right] = blurred
    
    return img

def draw_border(img, border, col=(0,255,0)):
    """Draw a border on an image given the border coordinates."""
    l, r, t, b = border
    cv2.line(img, (l, t), (l, b), col, 2)
    cv2.line(img, (l, t), (r, t), col, 2)
    cv2.line(img, (l, b), (r, b), col, 2)
    cv2.line(img, (r, t), (r, b), col, 2)

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

        self.error = False

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

    def filter1(self,hor_rat, ver_rat, min_ak, max_ak, min_ad, max_height):
        """Filters components based on height, horizontal ratio, and small area."""
        # Parameters for filter1:
        # min_ak = 20 # Top k elements with to calculate min allowed area
        min_ad = min_ad/10 # divider integer to get min allowed area of component
        # max_height = (self.img.shape[0]/35)*3 # max allowed height of component
        hor_rat = hor_rat/10 # max allowed horizontal ratio of component
        # max_ak = 4 # multiplier to get max allowed area of component
        ver_rat = ver_rat/10 # max allowed vertical ratio of component
        # Find minimum area under which we can delete components

        self.min_area = mean_top_k(self.area, min_ak)/min_ad
        if math.isnan(self.min_area):
            self.error = True
            return self.error
        else:
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

    def filter(self,file,hor_rat, ver_rat, min_ak, max_ak, min_ad, max_height):
        """Filters components."""
        self.borders = self.bounding_boxes()
        # cv2.imwrite('results/{}/components_borders0.png'.format(file), self.borders)
        # Filters based on height, horizontal ratio, vertical ratio, and very small area

        self.filter1(hor_rat, ver_rat, min_ak, max_ak, min_ad, max_height)
        if self.error == True:
            return 
        else:
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
            self.filter2()
            
            # Draw connected components after filter2
            self.filtered = np.zeros((self.img.shape))
            for i in range(len(self.allowed)):
                self.filtered[self.output == self.allowed[i] + 1] = 255

            # Draw bounding boxes after filter2
            self.borders = self.bounding_boxes()

            cv2.imwrite('results/{}/components_filtered2.png'.format(file), self.filtered)
            # cv2.imwrite('results/{}/components_borders2.png'.format(file), self.borders)


# Creates connected components
def connected_components(img,file,hor_rat, ver_rat, min_ak, max_ak, min_ad, max_height):
    """Create, visualize, filter, and return connected components."""
    # Create, filter, and return connected components
    components = Components(img)
    components.filter(file,hor_rat, ver_rat, min_ak, max_ak, min_ad, max_height)
    if components.error==True:
        return components.error
    else:
        return components

from utils import resize
from scipy.signal import find_peaks


def piecewise_linear(x, x0, y0, k1, k2):
    """Define a piecewise, lienar function with two line segments."""
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def getHorizontalProjectionProfileSmoothed(image):
    horizontal_projection = np.sum(image==255, axis = 1) 
    box = np.ones(40)/40
    smooth = np.convolve(horizontal_projection,box,mode='valid')
    
    return smooth
    

def gmm_clustering(cY, components,file,path,peak_thresh):
    """Uses GMM models to cluster text lines based on their y values."""
    from sklearn.mixture import GaussianMixture

    # Horizontal projection profile to determine lines
    comp_img = cv2.imread('./results/{}/components_filtered2.png'.format(file))
    comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2GRAY)
    hor_prof = getHorizontalProjectionProfileSmoothed(comp_img)
    peaks, _ = find_peaks(hor_prof,height=peak_thresh,distance=60)
    plt.rcParams["figure.autolayout"] = True
    plt.plot(hor_prof)
    plt.plot(peaks, hor_prof[peaks], 'x')
    # plt.savefig('results/{}/line_histogram.png'.format(file))
    plt.close()
    n = len(peaks)
    # n = no_lines
    if n == 0:
        print('Number of lines set: 0')
        return []
    gt_words_lines = get_gt_lines(path,file)
    print('Number of lines set: ', n)
    if n != len(gt_words_lines):
        return []
    # Perform analysis with determined number of components n
    gmm = GaussianMixture(n_components=n, random_state=0).fit(np.expand_dims(cY, 1))
    cluster_means = (gmm.means_.squeeze()).astype(np.int32)
    cluster_means.sort()
    return cluster_means


def line_clustering(components,file,path, peak_thresh):
    """Clusters components into horizontal lines."""

    # Organize and sort component data by y values
    c_area = components.area
    cX = components.x
    cY = components.y
    boundingRect = components.bounding_rect
    sorted_indices = cY.argsort(axis=0)
    c_area = c_area[sorted_indices]
    cX = cX[sorted_indices]
    cY = cY[sorted_indices]
    boundingRect = boundingRect[sorted_indices]
    mean_height = boundingRect[:, 3].mean()

    # Perform GMM analysis to determine lines based on y values
    cluster_means = gmm_clustering(cY, components,file,path, peak_thresh)
    if cluster_means == []:
        return []
    # Now that we've found the cluster y values, assign components to each cluster based on y
    component_clusters = np.zeros(len(components))
    component_clusters_min_dist = np.zeros(len(components))

    cluster_i = 0
    line_components = [[]]
    component_clusters = []
    for i in range(len(cY)):
        if cluster_i < len(cluster_means) - 1:
            if abs(cY[i] - cluster_means[cluster_i]) > abs(cluster_means[cluster_i + 1] - cY[i]):
                cluster_i += 1
                line_components.append([])
        
        line_components[-1].append(i)
        component_clusters.append(cluster_i)
    
    component_clusters = np.array(component_clusters)

    # Convert the 'sorted y' indices back to the original component indices
    for i, l in enumerate(line_components):
        sorter = np.argsort(sorted_indices)
        line_components[i] = sorter[np.searchsorted(sorted_indices,
                                                    np.array(l), sorter=sorter)]

    # Filter out lines with very little area
    # lines = [i for i, l in enumerate(line_components) if
    #                    components.area[l].sum() >= components.min_area*2]
    # line_components = [l for i, l in enumerate(line_components) if i in lines]
    # cluster_means = [m for i, m in enumerate(cluster_means) if i in lines]

    # Create display image
    keep_components = np.zeros((components.output.shape))
    for c in range(len(cluster_means)):
        for i in range(len(components)):
            if component_clusters[i] == c:
                keep_components[components.output == components.allowed[i] + 1] = 255

    for i, cc in enumerate(cluster_means):
        cv2.line(keep_components, (0, cluster_means[i]), (
            keep_components.shape[1], cluster_means[i]), 255, 3)

    return line_components

def crop_to_black(img):
    """Crops a 2D image, where each pixel is 0 or 255, to the black pixels."""
    black = np.argwhere(img == 0)
    t = black[:, 0].min()
    b = black[:, 0].max()
    l = black[:, 1].min()
    r = black[:, 1].max()
    return img[t:b, l:r]


def threshold_otsu(img, thresh_multiplier=None, color1=255, color2=0):
    """Threshold a greyscale image using otsu thresholding. Uses some erosion."""
    ret, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, ksize=(2, 2))
    thresh = cv2.erode(thresh, kernel)
    thresh[thresh >= 1] = color1
    thresh[thresh < 1] = color2
    return thresh


def threshold_mean(img, thresh_multiplier=0.95, color1=255, color2=0):
    """Threshold a greyscale image using mean thresholding."""
    mean = img.mean()
    ret = mean * thresh_multiplier
    img[img > ret] = color1
    img[img < ret] = color2
    return img


def threshold_multiple(img, thresh_func, count, thresh_multiplier=0.95, color1=255, color2=0):
    """
    Splits an image into multiple sections, thresholding these sections, and re-combines them
    into a final thresholded image. This is more robust to changes in light across the image.
    """
    w = int(np.round(img.shape[1]/(count + 1)))
    for i in range(count + 1):
        img[:, i*w:(i + 1)*w] = thresh_func(img[:, i*w:(i + 1)*w],
                                                   thresh_multiplier=thresh_multiplier,
                                                   color1=color1, color2=color2)
    return img


def threshold_multiple_line(img, thresh_func, page_width, thresh_multiplier=0.95, color1=255, color2=0):
    """
    Utilizes threshold_multiple, first determining an appropriate image split count
    for that line size. The wider the line, the larger the count.
    """
    count = int(np.round((img.shape[1]/page_width)*15))
    return threshold_multiple(img, thresh_func, count, thresh_multiplier=thresh_multiplier, color1=color1, color2=color2)


def remove_vertical_components(components, ind):
    """Removes vertically skinny components, which are often unwanted lines/artifacts in the image."""
    w = components.right[ind] - components.left[ind]
    h = components.bottom[ind] - components.top[ind]
    h_w_rat = 1.7
    # Return components with an acceptable h/w ratio
    return ind[np.argwhere(h/w < h_w_rat)[:, 0]]

def clean_line_thresh(img, consider_location=True, area_multiplier=1):
    """Cleans a thresholded image using morphing and connected components."""

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove smallest components with connected component analysis

    # Create components
    components = Components(255-img)

    # Intuition for consider_location: Small components near a y mean weighted by area are more
    # likely to be disconnected letter segments, small whereas components near the bottom/top of
    # the image are much more likely to be noise!
    if consider_location and np.sum(components.area) != 0:

        # Take a weighted mean of the y value, where the component area is the weight
        y_weighted_mean = np.average(components.y, weights=components.area)

        # Get each component's y distance from the weighted mean
        dist = np.abs(components.y - y_weighted_mean)

        # Squash this into a proportion
        dist = dist/max([y_weighted_mean, img.shape[0] - y_weighted_mean])
        
        min_area = mean_top_k(components.area, k=15)/8
        allowed = np.argwhere(((1 - dist)**2)*components.bounding_area > min_area * area_multiplier)[:, 0]
    else:

        min_area = mean_top_k(components.area, k=15)/3
        allowed = np.argwhere(components.bounding_area > min_area * area_multiplier)[:, 0]

    img = np.zeros((img.shape))
    for i in range(len(allowed)):
        img[components.output == allowed[i] + 1] = 255
    img = 255 - img
    
    return img


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


# Use gaps in lines to determine a suitable minimum gap between two words
def get_min_gap(lines, page_width):
    """
    Determines a minimum gap which exists between words. Under the right circumstances, gaps larger
    than this should be considered spaces.
    """
    widths = np.array([lines[i].right - lines[i].left for i in range(len(lines))])
    gaps_all = [lines[i].gaps for i in range(len(lines))]
    gaps_slanted_all = [lines[i].gaps_slanted for i in range(len(lines))]
    gaps_both_all = [lines[i].gaps_both for i in range(len(lines))]

    # Get line width proportion to page width and add 16%, which is about the max size of a border
    # This gives us how much of the page this line takes up
    line_width_proportions = widths/page_width+0.16
    # print('widths:',widths)
    # print('page_width:',page_width)
    # print('line_width_proportions:',line_width_proportions)
    # Multiplying these by the average words per line (10), gives us an expected word count
    #expected_words = line_width_proportions*10
    # Now, we'll adjust the expected words based on the text and space size
    
    # Generally, there is at most 11 words per line, so if the line proportion is 1, we'll take the
    # top 10 spaces for a full line
    min_gap = 0
    count = 0
    # print('gaps_all: ', gaps_both_all)
    for g, gaps in enumerate(gaps_both_all):
        if len(gaps) != 0:
            k = int(np.ceil(line_width_proportions[g] * 10) - 1)
            gaps = np.array(gaps)
            ranges = gaps[:, 1] - gaps[:, 0]
            ranges.sort()

            # Don't count lines with less than 3 expected words, since they may have just one word,
            # this would mess min_gap to count them!
            if k > 3:
                min_gap += ranges[-k:].mean()
                count += 1

    # No lines w/ words/gaps detected... assume all lines have one word, so use a massive min_gap
    if count == 0:
        return page_width

    # print('count: ',count)
    print('min gap: ', int(np.round(min_gap/count)))
    return int(np.round(min_gap/count))


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

    # Draw middles
    display_img = img.copy().astype(np.float32)
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)

    for m in middles:
        display_img = cv2.line(display_img, (m, 0), (m, display_img.shape[0]), (255, 0, 0), 2)

    for m in middles_slanted:
        display_img = cv2.line(display_img, (m, 0), (m, display_img.shape[0]), (0, 255, 0), 2)

    for m in middles_both:
        display_img = cv2.line(display_img, (m, 0), (m, display_img.shape[0]), (0, 0, 255), 2)

    # Merge the multiple analyses into one
    middles.extend(middles_slanted)
    middles.extend(middles_both)
    middles_merged = np.array(middles)
    middles_merged.sort()

    if len(middles_merged) == 0:
        return [], img, img

    merge_sum = middles_merged[0]
    merge_count = 1
    middles_final = []
    for i in range(1, len(middles_merged)):
        if middles_merged[i] - middles_merged[i - 1] < min_gap:
            merge_sum += middles_merged[i]
            merge_count += 1
        else:
            middles_final.append(int(np.round(merge_sum/merge_count)))
            merge_sum = middles_merged[i]
            merge_count = 1
    
    middles_final.append(int(np.round(merge_sum/merge_count)))  

    for c in middles_final:
        display_img = cv2.line(display_img, (c-4, 0), (c-4, display_img.shape[0]), 0, 2)
        display_img = cv2.line(display_img, (c+4, 0), (c+4, display_img.shape[0]), 0, 2)

    return middles_final, display_img, img


class Word():
    """
    Holds information about a word in the text.

    Attributes
    ----------
    left : int
        The left border of the word in the image.
    right : int
        The right border of the word in the image.
    top : int
        The top border of the word in the image.
    bottom : int
        The bottom border of the word in the image.
    words : list[np.ndarray]
        A list of images of the word.
    """

    def __init__(self, images, left, right, top, bottom):
        self.images = images
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

class Line():
    """
    Holds information about (and performs operations on) a line of text.

    Attributes
    ----------
    left : int
        The left border of the line in the image.
    right : int
        The right border of the line in the image.
    top : int
        The top border of the line in the image.
    bottom : int
        The bottom border of the line in the image.
    words : list[Word]
        A list of Word objects containing information about each word.

    Methods
    -------
    get_middles():
        Get space dividers (middles) for the line.
    
    crop_words():
        Segments a line image into word images.
    """

    def __init__(self, components, line, img,file):
        self.components = components

        # Remove artifact components which are easier to detect in the context of a line
        self.line = remove_vertical_components(components, line)

        if len(self.components.left[self.line]) == 0:
            self.valid = False
            return

        self.valid = True
        

        # Extract line bounding box information
        self.left = self.components.left[self.line].min()
        self.right = self.components.right[self.line].max()
        self.top = self.components.top[self.line].min()
        self.bottom = self.components.bottom[self.line].max()
        self.coords = [self.left,self.right,self.top,self.bottom]

        # Create connected components image
        self.comp_img = np.zeros(img.shape)
        for j in self.line:
            self.comp_img[self.components.output == self.components.allowed[j] + 1] = 255
        self.comp_img = cv2.bitwise_not(self.comp_img[self.top:self.bottom, self.left:self.right].astype(np.uint8))


        # Create a (multiple) thresholded image
        self.thresh_img = img[self.top:self.bottom, self.left:self.right]
        self.thresh_img = threshold_multiple_line(self.thresh_img, threshold_mean, img.shape[1])
        self.thresh_img = clean_line_thresh(self.thresh_img)

        # config['save_inter_func'](config, self.thresh_img, "line_thresh")

        # Calculate gaps for connected components and thresholded images
        self.gaps, self.gaps_slanted, self.gaps_both = get_gaps(self.comp_img)
        self.gaps_thresh, self.gaps_slanted_thresh, self.gaps_both_thresh = get_gaps(self.thresh_img)


    def get_middles(self, min_gap,file):
        '''Get space dividers (middles) for the line.'''
        # Get middles for components
        self.middles, display_img, thresh = get_middle(self.comp_img, self.gaps, self.gaps_slanted,
                                            self.gaps_both, min_gap)

        for m in self.middles:
            display_img = cv2.line(display_img, (m, 0), \
                                   (m, display_img.shape[0]), (0, 255, 0), 3)
        # config['save_inter_func'](config, display_img, "line_separated1")

        # Get middles for thresholded
        self.middles_thresh, display_img, thresh = get_middle(self.thresh_img, self.gaps_thresh,
                                                          self.gaps_slanted_thresh, self.gaps_both_thresh,
                                                          min_gap)

        for m in self.middles_thresh:
            display_img = cv2.line(display_img, (m, 0), \
                                   (m, display_img.shape[0]), (0, 255, 0), 3)
        # config['save_inter_func'](config, display_img, "line_separated2")


        # If the thresholded image doesn't have a line where the components image does, it should be removed
        # This is because the thresholded image has less missing text, so middles could created via missing text
        middles_final = get_middle_delete_unmatched(self.middles, self.middles_thresh, min_gap)

        display_img = thresh.copy()
        for m in middles_final:
            display_img = cv2.line(display_img, (m, 0), \
                                   (m, display_img.shape[0]), (0, 255, 0), 3)
        # config['save_inter_func'](config, display_img, "line_separated_final")

        return middles_final, thresh


    def crop_words(self, img, min_gap,file):
        '''Segments a line image into word images.'''
        self.middles_final, thresh = self.get_middles(min_gap,file)
        self.words = []

        # If there is no gaps, create just one word
        if len(self.middles_final) == 0:
            segments = [self.line]

        # Otherwise, separate the components into lines
        else:
        
            # Determine which components are in each segment
            x_line = self.components.x[self.line] - self.left
            self.middles_final = np.append(self.middles_final, img.shape[1])
            self.middles_final.sort()

            segments = [[] for i in range(len(self.middles_final))]
            for x_ind in range(len(x_line)):
                segments[np.argmax(self.middles_final > \
                                   x_line[x_ind])].append(self.line[x_ind])

        # Crop to the components in the line
        for j, s in enumerate(segments):
            left_seg = self.components.left[s]
            right_seg = self.components.right[s]
            top_seg = self.components.top[s]
            bottom_seg = self.components.bottom[s]

            if len(left_seg) > 0:
                l = left_seg.min()
                r = right_seg.max()
                t = top_seg.min()
                b = bottom_seg.max()

                # Create word images
                word1 = thresh[t - self.top:b - self.top, \
                               l - self.left:r - self.left]
                try:
                    word1 = crop_to_black(word1).astype(np.uint8)
                except ValueError:
                    continue

                word2 = img[t:b, l:r]
                word2 = threshold_multiple(word2, threshold_otsu, 4)
                word2 = clean_line_thresh(word2, consider_location=True, \
                                          area_multiplier=2).astype(np.uint8)

                # Recolor word images for SimpleHTR
                word1[word1 == 0] = 155
                word1[word1 == 255] = 232

                word2[word2 == 0] = 155
                word2[word2 == 255] = 232
                
                self.words.append(Word([word1, word2], l, r, t, b))

def bounding_boxes(img,left,right,top,bottom, width, height):
    """Draws bounding box for every component."""
    x = left + width//2
    borders = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(len(x)):
        draw_border(borders, (left[i], right[i], top[i], bottom[i]), col=255)

    # bounding_rect = np.stack([x, y, width, height], axis=1)
    return borders

def vertical_projections(sobel_image):
    #threshold the image.
    sum_of_rows = []
    for col in range(sobel_image.shape[1]-1):
        sum_of_rows.append(np.sum(sobel_image[:,col]))
    
    return sum_of_rows

def get_words_in_line(no_noise,img, components, line_components,file, min_gap):
    '''Segments a line image into word images.'''
    # Process each line
    lines = []
    for i, line in enumerate(line_components):
        line_obj = Line(components, line, img,file)
        if line_obj.valid:
            lines.append(line_obj)

    # Use gaps in lines to determine a suitable minimum gap between two words
    # min_gap = get_min_gap(lines, img.shape[1])

    # Crop lines into words
    words = []

    for i in range(len(lines)):
        lines[i].crop_words(img, min_gap,file)

    return lines

def get_gt_BB_len(path,file):
    return len(get_word_vector(path,file))

def main_process(path, file, holes, t1, hor_rat, ver_rat, min_ak, max_ak, min_ad, max_height, peak_thresh, min_gap):
    if (os.path.exists('./results/{}/'.format(file))) == False:
        os.mkdir('./results/{}/'.format(file))

    stream = open(u'{}{}.jpg'.format(path,file), "rb")
    bytes = bytearray(stream.read())
    numpyarr = np.asarray(bytes,dtype=np.uint8)

    image = cv2.imdecode(numpyarr, cv2.IMREAD_UNCHANGED)
    page_image,s_points,t_points = page_segment(image)

    # Remove page holes if any
    if holes == 'y':
        gray = page_hole_removal(page_image)
    else:
        gray = cv2.cvtColor(page_image,cv2.COLOR_BGR2GRAY)

    def noise_removal(image,thresh_param,kernel_size):
        thresh, image = cv2.threshold(image, thresh_param, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    no_noise = noise_removal(gray,t1, 1)

    row, col = no_noise.shape[:2]
    bottom = no_noise[row - 2:row, 0:col]
    bordersize = 1
    no_noise = cv2.copyMakeBorder(
    no_noise,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=0
    )
    cv2.floodFill(no_noise, None, (0,0), 255)
    y, x = no_noise.shape
    no_noise = no_noise[1:y-1, 1:x-1]

    edges = cv2.Canny(no_noise, 255, 255/3)
    
    components = connected_components(edges,file,hor_rat, ver_rat, min_ak, max_ak, min_ad, max_height)
    if type(components) == bool:
        return 0
    line_components = line_clustering(components,file,path, peak_thresh)
    if line_components == []:
        return 0
    lines = get_words_in_line(no_noise,gray, components, line_components,file, min_gap)

    lines_image = page_image.copy()
    words_image = page_image.copy()

    word_coords = []

    # Creates bounding boxes around lines/words and saves images
    for i,line in enumerate(lines):
        line_image = page_image.copy()
        cv2.rectangle(lines_image, (int(line.left),int(line.top)), (int(line.right),int(line.bottom)),(0,255,0),4)
        for j, word in enumerate(line.words):
            cv2.rectangle(words_image, (int(word.left),int(word.top)), (int(word.right), int(word.bottom)), (0,255,0),4)
            word_pos = [int(word.left),int(word.right), int(word.top), int(word.bottom)]
            word_coords.append([i,word_pos])

    # print('word_coords: ', word_coords)
    no_of_lines = len(lines)
    predicted_lines = [0]*no_of_lines
    for i,coords in word_coords:
        predicted_lines[i] += 1

    cv2.imwrite('results/{}/lines_img.png'.format(file), cv2.cvtColor(lines_image,cv2.COLOR_BGR2RGB))
    cv2.imwrite('results/{}/words_img.png'.format(file), cv2.cvtColor(words_image,cv2.COLOR_BGR2RGB))

    return predicted_lines

