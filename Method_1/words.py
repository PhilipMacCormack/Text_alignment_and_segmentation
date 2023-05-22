import numpy as np
import cv2
import matplotlib.pyplot as plt
from Method_1.connectedComponents import Components, mean_top_k, draw_border
from Method_1.utils import resize, page_resize, filter1, mean_top_k, get_iou
from Method_1.selectinwindow import *

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
    # import connectedComponents as ccp

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
    # print('min gap: ', int(np.round(min_gap/count)))
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

class Line_added():
    def __init__(self,components, x1,x2,y1,y2, img, file,line_comps):
        self.components = components
        self.line = remove_vertical_components(components, line_comps)

        # filter1(self.linecomps, hor_rat = 150/10, ver_rat = 27/10, min_ak = 69, max_ak = 2, min_ad = 22/10, max_height = 158)
        self.valid = True
        self.left = x1
        self.right = x2
        self.top = y1
        self.bottom = y2

        # Create connected components image
        self.comp_img = np.zeros(img.shape)
        for j in self.line:
            self.comp_img[self.components.output == self.components.allowed[j] + 1] = 255
        self.comp_img = cv2.bitwise_not(self.comp_img[self.top:self.bottom, self.left:self.right].astype(np.uint8))

        # Create a (multiple) thresholded image
        self.thresh_img = img[self.top:self.bottom, self.left:self.right]
        self.thresh_img = threshold_multiple_line(self.thresh_img, threshold_mean, img.shape[1])
        self.thresh_img = clean_line_thresh(self.thresh_img)

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
            # cv2.imwrite('results/{}/line_separated_final.png'.format(file), display_img)

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

class Line():
    """
    Attributes
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
        self.line_components = line
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
            # cv2.imwrite('results/{}/line_comp{}.png'.format(file,j), self.comp_img)
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
            # cv2.imwrite('results/{}/line_separated_final.png'.format(file), display_img)

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

def get_words_in_line(page_img,no_noise, components, line_components,file, min_gap, scale_percent, pre_min_gap):
    '''Segments a line image into word images.'''
    # Process each line
    lines = []
    line_coords = []
    line_img = page_img.copy()
    # print('line components: ', line_components)
    # print('components: ', components)
    for i, line in enumerate(line_components):
        line_obj = Line(components, line, no_noise,file)
        if line_obj.valid:
            lines.append(line_obj)
            line_coords.append([int(line_obj.left),int(line_obj.right), int(line_obj.top), int(line_obj.bottom)])
            cv2.rectangle(line_img, (int(line_obj.left),int(line_obj.top)), (int(line_obj.right),int(line_obj.bottom)),(0,255,0),4)
    # print('len(lines): ',len(lines))
    def min_gap_visualise(no_noise,file, min_gap):
        for i in range(len(lines)):
            lines[i].crop_words(no_noise, min_gap,file)
        for line in lines:
            for j, word in enumerate(line.words):
                cv2.rectangle(words_image, (int(word.left),int(word.top)), (int(word.right), int(word.bottom)), (0,255,0),4)
        return words_image

    def inside(bl, tr, p) :
            if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
                return True
            else :
                return False

    remove_line_boxes = []
    removed_line_comps = []
    resized_lineimg = resize(line_img, scale_percent)
    
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
                    removed_line_comps.append([lines[selected_inside_ind].line_components, lines[selected_inside_ind].left, lines[selected_inside_ind].right, lines[selected_inside_ind].top, lines[selected_inside_ind].bottom])
                    del line_coords[selected_inside_ind]
                    del lines[selected_inside_ind]
                else:
                    removed_line_comps.append([lines[inside_inds[0]].line_components, lines[inside_inds[0]].left, lines[inside_inds[0]].right, lines[inside_inds[0]].top, lines[inside_inds[0]].bottom])
                    del line_coords[inside_inds[0]]
                    del lines[inside_inds[0]]

    # print(removed_line_comps)
    new_line_image = page_img.copy()
    for coord in line_coords:
        cv2.rectangle(new_line_image,(coord[0],coord[2]),(coord[1],coord[3]), (0,255,0),4)

    resized2 = resize(new_line_image, scale_percent)
    import sys
    sys.setrecursionlimit(10 ** 9)
    wName = 'Add line bounding-boxes'
    rectI = DragRectangle(resized2, wName, new_line_image.shape[0], new_line_image.shape[1])
    cv2.namedWindow(rectI.wname)
    cv2.setMouseCallback(rectI.wname, dragrect, rectI)
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
                    if len(remove_line_boxes) == 1:
                        line = Line_added(components, x1,x2,y1,y2, no_noise, file, removed_line_comps[0][0])
                        lines.append(line)
                    if len(remove_line_boxes) > 1:
                        ious = []
                        for remove_coord in removed_line_comps:
                            ious.append(get_iou([x1,x2,y1,y2], [remove_coord[1],remove_coord[2],remove_coord[3],remove_coord[4]]))
                        max_iou_index = ious.index(max(ious))
                        line_comps = removed_line_comps[max_iou_index][0]
                        line = Line_added(components, x1,x2,y1,y2, no_noise, file, line_comps)
                        lines.append(line)
                    print('Line box registered.')
                rectI.reset()
        if key == 8:
            if cache_images != []:
                rectI.image = cache_images[-1]
                new_line_image = cache_line_images[-1]
                del cache_images[-1]
                del line_coords[-1]
                del lines[-1]
                del cache_line_images[-1]

        if key == ord('q'):
            break

    # cv2.imwrite('results/{}/corrected_line_image.png'.format(file), cv2.cvtColor(new_line_image,cv2.COLOR_BGR2RGB))

    cv2.destroyAllWindows()
    line_coords.sort(key = lambda x: x[3])
    lines.sort(key = lambda x: x.top)

    if min_gap is None:
        def nothing(x):
            pass
        cv2.namedWindow('mingap_win')
        cv2.createTrackbar('min_gap','mingap_win',pre_min_gap,65,nothing)
        print("Choose a suitable minimum gap for the document. Press 'Q' when you are satisfied.")
        while(1):
            words_image = no_noise.copy()
            word_img = min_gap_visualise(words_image, file, cv2.getTrackbarPos('min_gap','mingap_win'))
            word_img_resized = resize(word_img, scale_percent)
            cv2.imshow('min_gap_win', word_img_resized)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
        min_gap = cv2.getTrackbarPos('min_gap','mingap_win')
        cv2.destroyAllWindows()

    print('Min gap set to:', min_gap)
    # Crop lines into words
    words = []

    # from skimage.filters import sobel
    
    for i in range(len(lines)):
        # line_img = no_noise[lines[i].top-5:lines[i].bottom+5, lines[i].left-5:lines[i].right+5]
        
    #     #border creation to fill border hitting contours
    #     row, col = line_img.shape[:2]
    #     bottom = line_img[row - 2:row, 0:col]
    #     bordersize = 1
    #     line_img = cv2.copyMakeBorder(
    #     line_img,
    #     top=bordersize,
    #     bottom=bordersize,
    #     left=bordersize,
    #     right=bordersize,
    #     borderType=cv2.BORDER_CONSTANT,
    #     value=0
    #     )
    #     cv2.floodFill(line_img, None, (0,0), 255)

    #     #removing the created borders
    #     y, x = line_img.shape
    #     line_img = line_img[1:y-1, 1:x-1]

    #     # line_img_blur = cv2.medianBlur(line_img,5)

    #     sobel_line = sobel(line_img)
    #     vpp = vertical_projections(sobel_line)
    #     threshold = 1
    #     space_count = 0
    #     space_gap = 30
    #     gapp = 0
    #     for ind,vppv in enumerate(vpp):
    #         if gapp == int(space_gap/2):
    #             x_mid = ind
    #         if gapp >= space_gap:
    #             if vppv <= threshold:
    #                 cv2.line(line_img, (x_mid,0),(x_mid,line_img.shape[1]), (0,255,0),2)
    #                 continue
    #         if vppv >= threshold:
    #             gapp = 0
    #         if vppv < threshold:
    #             gapp += 1
    #             if gapp >= space_gap:
    #                 space_count += 1
    #     print('spaces found in line ',i,': ', space_count)


        # plt.plot(vpp)
        # plt.title(str(i))
        # plt.show()

        # cv2.imwrite('results/{}/line{}.png'.format(file, i), line_img)

        # box_img = bounding_boxes(line_img, left, right, top, bottom, width, height)
        # cv2.imwrite('results/{}/line_conncomp{}.png'.format(file, i), box_img)

        lines[i].crop_words(no_noise, min_gap,file)

    return lines, line_coords, min_gap