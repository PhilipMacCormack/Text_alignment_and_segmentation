import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from Method_2.page_segment import page_segment
from Method_2.hole_remove import page_hole_removal
from Method_2.connectedComponents import *
from sklearn.cluster import KMeans
from Method_2.utils import resize
from Method_2.selectinwindow import *
from Method_2.selectinwindow import *

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

# Use gaps in lines to determine a suitable minimum gap between two words
def get_min_gap(line_coord, gaps, gaps_slanted, gaps_both, page_width,line_img):
    """
    Determines a minimum gap which exists between words. Under the right circumstances, gaps larger
    than this should be considered spaces.
    """
    width = line_coord[3] - line_coord[2]
    # Get line width proportion to page width and add 16%, which is about the max size of a border
    # This gives us how much of the page this line takes up
    line_width_proportion = width/page_width+0.16
    print('width:',width)
    print('page_width:',page_width)
    print('line_width_proportions:',line_width_proportion)
    # Multiplying these by the average words per line (10), gives us an expected word count
    #expected_words = line_width_proportions*10
    # Now, we'll adjust the expected words based on the text and space size
    
    # Generally, there is at most 11 words per line, so if the line proportion is 1, we'll take the
    # top 10 spaces for a full line
    min_gap = 0
    count = 0
    if len(gaps) != 0:
        k = int(np.ceil(line_width_proportion * 10) - 1)
        gaps = np.array(gaps)
        print('gaps: ', gaps)
        ranges = gaps[:, 1] - gaps[:, 0]
        ranges.sort()
        print('ranges: ', ranges)


        # Don't count lines with less than 3 expected words, since they may have just one word,
        # this would mess min_gap to count them!
        if k > 3:
            min_gap += ranges[-k:].mean()
            count += 1
    test = line_img.copy()
    for gap in gaps:
        cv2.line(test, (gap[0],line_coord[2]), (gap[1],line_coord[3]),(0,255,0),2)
    cv2.imshow('gaps', test)
    cv2.waitKey(0)
    # No lines w/ words/gaps detected... assume all lines have one word, so use a massive min_gap
    if count == 0:
        return page_width

    print('count: ',count)
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

def line_segmentation(no_noise, components, scale_percent,page_image,file):
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
    cv2.imwrite('results/{}/line_y.png'.format(file), testimg)

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

def GMM_line_segmentation(no_noise, components, scale_percent,page_image,file,path):
    line_components = line_clustering(components, file, path)
    line_coords = []
    for line in line_components:
        x1 = components.left[line].min()
        x2 = components.right[line].max()
        y1 = components.top[line].min()
        y2 = components.bottom[line].max()
        line_coords.append([x1,x2,y1,y2])
    line_seg = cv2.cvtColor(no_noise, cv2.COLOR_GRAY2RGB)
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

def word_segmentation(line_coords, no_noise, file, min_gap):
    word_coords = []
    for i,coord in enumerate(line_coords):
        line_no = i+1
        line_img = no_noise[coord[2]:coord[3], coord[0]:coord[1]]
        line_img_temp = no_noise[coord[2]-10:coord[3]+10, coord[0]-10:coord[1]+10]
        # print('line :', i+1)

        line_img_edges = cv2.Canny(line_img, 255, 255/3)
        components = connected_components(line_img_edges, file)
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

        # min_gap = get_min_gap(coord, gaps, gaps_slanted, gaps_both, page_image.shape[1],line_img)
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

## ------------- MAIN ------------------------------------------------------------------------------------

# path = '../../../../Data/Labours_Memory/export_job_2576375/776034/Örsundsbro,_Giresta_avd_025/'
# # path = '../../../../Data/Labours_Memory/export_job_2576394/788025/Härkeberga_avd_429_Årsb_Lantarb/'
# # file = 'fac_03008_verksamhetsberattelse_1952_sid-03_1'
# file = 'fac_03008_arsberattelse_1931'
# t1 = 212
# min_gap = 25
# stream = open(u'{}{}.jpg'.format(path,file), "rb")
# bytes = bytearray(stream.read())
# numpyarr = np.asarray(bytes,dtype=np.uint8)
# image = cv2.imdecode(numpyarr, cv2.IMREAD_UNCHANGED)
# # Remove borders from document
# page_image,s_points,t_points = page_segment(image)
# # gray = cv2.cvtColor(page_image,cv2.COLOR_BGR2GRAY)
# gray = page_hole_removal(page_image)
# def noise_removal(image,thresh_param,kernel_size):
#     thresh, image = cv2.threshold(image, thresh_param, 255, cv2.THRESH_BINARY)
#     kernel = np.ones((kernel_size,kernel_size), np.uint8)
#     image = cv2.dilate(image, kernel, iterations=1)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
#     image = cv2.medianBlur(image, 3)
#     return image
# no_noise = noise_removal(gray,t1, 1)

# # Remove noise at borders of image
# row, col = no_noise.shape[:2]
# bottom = no_noise[row - 2:row, 0:col]
# bordersize = 1
# no_noise = cv2.copyMakeBorder(
# no_noise,
# top=bordersize,
# bottom=bordersize,
# left=bordersize,
# right=bordersize,
# borderType=cv2.BORDER_CONSTANT,
# value=0
# )
# cv2.floodFill(no_noise, None, (0,0), 255)
# y, x = no_noise.shape
# no_noise = no_noise[1:y-1, 1:x-1]

# edges = cv2.Canny(no_noise, 255, 255/3)
# components = connectedComponents.connected_components(edges,file)

# # comp_img = cv2.imread('./results/fac_03008_arsberattelse_1931/components_filtered2.png')
# # comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2GRAY)
# line_coords = line_segmentation(no_noise, components,20,page_image,file,path)
# word_segmentation(line_coords,no_noise,file, min_gap)