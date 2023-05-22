import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import xmltodict
from screeninfo import get_monitors

from Method_2.page_segment import page_segment
from Method_2.hole_remove import page_hole_removal
from Method_2.connectedComponents import *
from Method_2.segmentation import GMM_line_segmentation, word_segmentation, line_segmentation
from Method_2.txt_ascii_read import get_word_vector
from Method_2.xml_word_coords import word_bb_coords
from Method_2.xml_word_coords import iam_word_bb_coords
from Method_2.utils import *
from Method_2.create_xml_output import xml_output
from Method_2.iou_metric import iou_metric
from Method_2.read_alignment_file import alignment_read
from Method_2.selectinwindow import *

def method_2(path,file,read_params,holes,transcribe_or_gt,pre_t1,min_gap):
    # Get file extension of image
    if os.path.isfile('{}{}.jpg'.format(path,file)):
        extension = '.jpg'

    elif os.path.isfile('{}{}.png'.format(path,file)):
        extension = '.png'  

    # Get height of monitor
    monitors = []
    for m in get_monitors():
        if len(get_monitors()) == 1:
            height = m.height
        else:
            monitors.append(m.height)
    if monitors != []:
        height = max(monitors) - 50

    # Create folder for storing results in for specific image
    print('File: {}'.format(file))
    if (os.path.exists('./results/{}/'.format(file))) == False:
        os.mkdir('./results/{}/'.format(file))

    # Check if bayesian optimisation has been done to fetch parameters
    if read_params:
        try:
            with open('../Bayesian_testing/results/bayesian_results_method_2/{}/best_params.xml'.format(file),'r',encoding='utf8') as f:
                data = f.read()
                dict_data = xmltodict.parse(data)
                target = float(dict_data['alto']['Target'])
                if target > 0:
                    t1 = int(dict_data['alto']['Parameters']['@t1'])
                    min_gap = int(dict_data['alto']['Parameters']['@min_gap'])
                    print('Parameters from Bayesian optimisation read,  t1={}   min_gap={}'.format(t1,min_gap))
                else:
                    t1 = None
                    min_gap = None
        except:
            print('Error: Could not read parameters from Bayesian optimisation.')
            t1 = None
            min_gap = None
    else:
        t1 = None

    # read image from path and file name
    stream = open(u'{}{}{}'.format(path,file, extension), "rb")
    bytes = bytearray(stream.read())
    numpyarr = np.asarray(bytes,dtype=np.uint8)
    image = cv2.imdecode(numpyarr, cv2.IMREAD_UNCHANGED)

    # Remove borders from document
    page_image,s_points,t_points,M = page_segment(image)

    # get scale percent to fit image to monitor size
    orig_height = page_image.shape[0]
    height_check_img = page_resize(page_image, height)
    new_height = height_check_img.shape[0]
    scale_percent = (new_height/orig_height)*100
    print('image scaled to screen height, scale percent: ', scale_percent)

    save_page = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('results/{}/page_img.png'.format(file), save_page)

    # Remove page holes if any
    if holes:
        gray = page_hole_removal(page_image)
    else:
        gray = cv2.cvtColor(page_image,cv2.COLOR_BGR2GRAY)

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

    if t1 is None:
        cv2.namedWindow('noise_remove')
        cv2.createTrackbar('t1','noise_remove',pre_t1,235,nothing)
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

    # Remove noise at borders of image
    row, col = no_noise.shape[:2]
    bottom = no_noise[row - 2:row, 0:col]
    bordersize = 1
    no_noise = cv2.copyMakeBorder(no_noise,top=bordersize,bottom=bordersize,left=bordersize,right=bordersize,borderType=cv2.BORDER_CONSTANT,value=0)
    cv2.floodFill(no_noise, None, (0,0), 255)
    y, x = no_noise.shape
    no_noise = no_noise[1:y-1, 1:x-1]

    edges = cv2.Canny(no_noise, 255, 255/3)

    cv2.imwrite('results/{}/no_noise_img.png'.format(file), no_noise)
    # cv2.imwrite('results/{}/edges_img.png'.format(file), edges)

    #Segment image into lines & words
    components = connected_components(edges,file)

    # line_coords = GMM_line_segmentation(no_noise, components, scale_percent,page_image, file, path)  # GMM LINE SEGMENTATION METHOD

    line_coords = line_segmentation(no_noise, components, scale_percent, page_image, file) # MY LINE SEGMENTATION METHOD
    word_coords = word_segmentation(line_coords, no_noise, file, min_gap) # MY WORD SEGMENTATION

    # line_components = lineClustering.line_clustering(components,file,path)
    # lines, line_coords = words.get_words_in_line(page_image,no_noise, components, line_components,file, min_gap, scale_percent)



    lines_image = page_image.copy()
    words_image = page_image.copy()

    if (os.path.exists('./results/{}/lines'.format(file))) == False:
            os.mkdir('./results/{}/lines'.format(file))

    line_images = []
    line_Ys =[]
    line_word_image = page_image.copy()
    # Creates bounding boxes around lines and saves images
    for i,line in enumerate(line_coords):
        line_image = page_image.copy()
        cropped_line = line_image[int(line[2]):int(line[3]), int(line[0]):int(line[1])]
    #     line_images.append(cropped_line)
        cv2.rectangle(lines_image, (int(line[0]),int(line[2])), (int(line[1]),int(line[3])),(0,255,0),4)
        cv2.rectangle(line_word_image, (int(line[0]),int(line[2])), (int(line[1]),int(line[3])),(255,0,0),4)
        cv2.imwrite('results/{}/lines/line{}.png'.format(file,i+1),cv2.cvtColor(cropped_line,cv2.COLOR_BGR2RGB))    
        # if (os.path.exists('./results/{}/lines/line{}'.format(file,i+1))) == False:
    #         os.mkdir('./results/{}/lines/line{}'.format(file,i+1))
        line_Ys.append(int(abs((line[3]+line[2])/2)))
    # for i,line_ob in enumerate(lines):
    #     # if line_coords[i][0] == 'new':
    #     #     line_no += 1
    #     # if line_coords[i][0] != 'new':
    #     #     avg_y = 0
    #     for j, word in enumerate(line_ob.words):
    #         cv2.rectangle(words_image, (int(word.left),int(word.top)), (int(word.right), int(word.bottom)), (0,255,0),4)
    #         cv2.rectangle(line_word_image, (int(word.left),int(word.top)), (int(word.right), int(word.bottom)), (0,255,0),4)
    #         word_pos = [int(word.left),int(word.right), int(word.top), int(word.bottom)]
    #         word_y = int((int(word.top)+int(word.bottom)) / 2)
    #         min_y_line = line_Ys.index(min(line_Ys, key=lambda x:abs(x-word_y))) + 1
    #         word_coords.append([min_y_line,word_pos])
    #         # if line_coords[i][0] != 'new':
    #         #     avg_y += (word.bottom+word.top)/2
    #     # if line_coords[i][0] != 'new':
    #     #     line_Ys.append(avg_y/len(line_ob.words))

    for coord in word_coords:
        cv2.rectangle(words_image, (coord[1][0],coord[1][2]),(coord[1][1],coord[1][3]),(0,255,0),2)

    resized_wordimg = resize(words_image, scale_percent)
    resized_linewordimg = resize(line_word_image, scale_percent)

    def inside(bl, tr, p) :
                if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
                    return True
                else :
                    return False

    remove_word_boxes = []
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            remove_word_boxes.append([x/(scale_percent/100),y/(scale_percent/100)])
            cv2.circle(resized_wordimg, (x,y),4,(0,0,255), -20)
            cv2.imshow('remove_win', resized_wordimg)

    print("Left click the green word boxes you want to correct to mark them. Press 'Q' when you are satisfied.")
    while(1):
        cv2.imshow('remove_win', resized_wordimg)
        cv2.setMouseCallback('remove_win', click_event)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    if remove_word_boxes != []:
        for point in remove_word_boxes:
            inside_areas = []
            inside_inds = []
            for ind, prev_box in enumerate(word_coords):
                if inside((prev_box[1][0],prev_box[1][2]),(prev_box[1][1],prev_box[1][3]),(point[0],point[1])):
                    inside_areas.append((prev_box[1][1]-prev_box[1][0])*(prev_box[1][3]-prev_box[1][2]))
                    inside_inds.append(ind)
            if inside_inds != []:
                # print('inside_inds: ', inside_inds)
                # print('inside_areas:', inside_areas)
                if len(inside_areas) > 1:
                    selected_inside_val = inside_areas.index(min(inside_areas))
                    # print('selected_inside_val: ', selected_inside_val)
                    selected_inside_ind = inside_inds[selected_inside_val]
                    del word_coords[selected_inside_ind]
                else:
                    del word_coords[inside_inds[0]]

    def get_iou(pred_word_bb, word_bb):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """
        # determine the coordinates of the intersection rectangle
        x_left = max(pred_word_bb[1][0], word_bb[1])
        y_top = max(pred_word_bb[1][2], word_bb[3])
        x_right = min(pred_word_bb[1][1], word_bb[2])
        y_bottom = min(pred_word_bb[1][3], word_bb[4])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        # compute the area of both AABBs
        bb1_area = (pred_word_bb[1][1] - pred_word_bb[1][0]) * (pred_word_bb[1][3] - pred_word_bb[1][2])
        bb2_area = (word_bb[2] - word_bb[1]) * (word_bb[4] - word_bb[3])
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    new_words_image = page_image.copy()
    for coord in word_coords:
        cv2.rectangle(new_words_image,(coord[1][0],coord[1][2]),(coord[1][1],coord[1][3]),(0,255,0),4)

    resized2 = resize(new_words_image, scale_percent)
    sys.setrecursionlimit(10 ** 9)
    wName = 'Add bounding-boxes'
    rectI = DragRectangle(resized2, wName, new_words_image.shape[0], new_words_image.shape[1])
    cv2.namedWindow(rectI.wname)
    cv2.setMouseCallback(rectI.wname, dragrect, rectI)
    print("Add new box by clicking & dragging, press & hold 'Enter' when you are satisfied with a box. Press 'Q' when you are finished.")
    print("Box-Autocorrection enabled. Press 'm' to toggle on/off.")
    box_auto_corr = True
    cache_images = []
    cache_word_images = []
    while(1):
        # display the image
        cv2.imshow(wName, rectI.image)
        key = cv2.waitKey(1)
        if key == ord('m'):
            if box_auto_corr == True:
                print("Box-Autocorrection disabled.")
                box_auto_corr = False
            else:
                print("Box-Autocorrection enabled.")
                box_auto_corr = True
        if key == 13:
            if box_auto_corr == True:
                y_diffs = []
                x1 = int(rectI.outRect.x/(scale_percent/100))
                x2 = int(rectI.outRect.x/(scale_percent/100)+rectI.outRect.w/(scale_percent/100))
                y1 = int(rectI.outRect.y/(scale_percent/100))
                y2 = int(rectI.outRect.y/(scale_percent/100)+rectI.outRect.h/(scale_percent/100))
                if (y1+y2)/2 == 0.0:
                    pass
                else:
                    height = (y2-y1)*(2/3)
                    width = (x2-x1)*(1/3)
                    area = (y2-y1)*(x2-x1)
                    y1_big = int(y1 - (height/2))
                    y2_big = int(y2 + (height/2))
                    x1_big = int(x1 - (width/2))
                    x2_big = int(x2 + (width/2))
                    word_img = no_noise[y1_big:y2_big,x1_big:x2_big]
                    edge_word_img = cv2.Canny(word_img, 255, 255/3)
                    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_word_img, connectivity=8)
                    word_part = []
                    for i in range(1,numLabels):
                        comp_area = stats[i,cv2.CC_STAT_AREA]
                        x1_comp = stats[i,cv2.CC_STAT_LEFT]
                        x2_comp = stats[i,cv2.CC_STAT_WIDTH] + stats[i,cv2.CC_STAT_LEFT]
                        y1_comp = stats[i,cv2.CC_STAT_TOP]
                        y2_comp = stats[i,cv2.CC_STAT_TOP] + stats[i,cv2.CC_STAT_HEIGHT]
                        iou = get_iou([0,[0,word_img.shape[1],0,word_img.shape[0]]],[0,x1_comp,x2_comp,y1_comp,y2_comp])
                        if iou > 0.01:
                            word_part.append(i)
                    x1_new = 100000
                    x2_new = 0
                    y1_new = 100000
                    y2_new = 0
                    for part in word_part:
                        if x1_new > stats[part,cv2.CC_STAT_LEFT]:
                            x1_new = stats[part,cv2.CC_STAT_LEFT]
                        if y1_new > stats[part,cv2.CC_STAT_TOP]:
                            y1_new = stats[part,cv2.CC_STAT_TOP]
                        if x2_new < stats[part,cv2.CC_STAT_WIDTH] + stats[part,cv2.CC_STAT_LEFT]:
                            x2_new = stats[part,cv2.CC_STAT_WIDTH] + stats[part,cv2.CC_STAT_LEFT]
                        if y2_new < stats[part,cv2.CC_STAT_HEIGHT] + stats[part,cv2.CC_STAT_TOP]:
                            y2_new = stats[part,cv2.CC_STAT_HEIGHT] + stats[part,cv2.CC_STAT_TOP]
                    new_orig_x1 = x1_big+x1_new
                    new_orig_x2 = x1_big+x2_new
                    new_orig_y1 = y1_big+y1_new
                    new_orig_y2 = y1_big+y2_new

                    for i in line_Ys: # line_Ys = [y-coordinate, ...]
                        y_diff = abs(i-((y1+y2)/2))
                        y_diffs.append(y_diff)
                    min_y_diff = (min(y_diffs))
                    min_y_ind = y_diffs.index(min_y_diff) + 1
                    new_box = [min_y_ind, [new_orig_x1, new_orig_x2, new_orig_y1, new_orig_y2]]
                    if new_box not in word_coords: # Prevents multiple entries of same box
                        cache_images.append(rectI.image.copy())
                        cache_word_images.append(new_words_image.copy())
                        cv2.rectangle(new_words_image,(new_orig_x1,new_orig_y1),(new_orig_x2,new_orig_y2), (0,255,0),4)
                        cv2.rectangle(rectI.image, (int(new_orig_x1*(scale_percent/100)), int(new_orig_y1*(scale_percent/100))), (int(new_orig_x2*(scale_percent/100)), int(new_orig_y2*(scale_percent/100))), (0,255,0),1)
                        word_coords.append(new_box)
                        print('Box on line {} registered.'.format(min_y_ind))
                    rectI.reset()
            if box_auto_corr == False:
                y_diffs = []
                x1 = int(rectI.outRect.x/(scale_percent/100))
                x2 = int(rectI.outRect.x/(scale_percent/100)+rectI.outRect.w/(scale_percent/100))
                y1 = int(rectI.outRect.y/(scale_percent/100))
                y2 = int(rectI.outRect.y/(scale_percent/100)+rectI.outRect.h/(scale_percent/100))
                if (y1+y2)/2 == 0.0:
                    pass
                else:
                    for i in line_Ys: # line_Ys = [y-coordinate, ...]
                        y_diff = abs(i-((y1+y2)/2))
                        y_diffs.append(y_diff)
                    min_y_diff = (min(y_diffs))
                    min_y_ind = y_diffs.index(min_y_diff) + 1
                    new_box = [min_y_ind, [x1,x2,y1,y2]]
                    if new_box not in word_coords: # Prevents multiple entries of same box
                        cache_images.append(rectI.image.copy())
                        cache_word_images.append(new_words_image.copy())
                        cv2.rectangle(new_words_image,(x1,y1),(x2,y2), (0,255,0),4)
                        cv2.rectangle(rectI.image, (int(x1*(scale_percent/100)), int(y1*(scale_percent/100))), (int(x2*(scale_percent/100)), int(y2*(scale_percent/100))), (0,255,0),1)
                        word_coords.append(new_box)
                        print('Box on line {} registered.'.format(min_y_ind))
                    rectI.reset()
        if key == 8:
            if cache_images != []:
                rectI.image = cache_images[-1]
                new_words_image = cache_word_images[-1]
                del cache_images[-1]
                del word_coords[-1]
                del cache_word_images[-1]

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # CREATE WORD AND LINE IMAGES
    cv2.imwrite('results/{}/lines_img.png'.format(file), cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB))
    if remove_word_boxes != []:
        cv2.imwrite('results/{}/words_corrected_img.png'.format(file), cv2.cvtColor(new_words_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite('results/{}/words_img.png'.format(file), cv2.cvtColor(words_image, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite('results/{}/words_img.png'.format(file), cv2.cvtColor(words_image, cv2.COLOR_BGR2RGB))

    for i,line in enumerate(line_coords):
        line_image = page_image.copy()
        line_word_boxes = []
        for word in word_coords:
            if word[0] == i+1:
                line_word_boxes.append(word)
        line_word_boxes.sort(key = lambda x: x[1])
        for j,box in enumerate(line_word_boxes):
                cropped_word = cv2.cvtColor(page_image[box[1][2]:box[1][3], box[1][0]:box[1][1]], cv2.COLOR_BGR2RGB)
                cv2.imwrite('results/{}/lines/line{}/word{}_{}.png'.format(file,box[0],box[0],j+1),cropped_word)

    no_of_lines = len(line_coords)
    # print('Predicted number of words: ', len(word_coords))
    # print('corrected word_coords: ', word_coords)

    while True:
        if transcribe_or_gt == None:
            transcribe_or_gt = input("Do you want to transcribe the document, or do you want to align from ground truth (1/2/3)? \n (1) Transcribe. \n (2) Align from transcript. \n (3) Align from box coordinates. \n Answer: ")
        # ALIGN FROM GROUND TRUTH
        if transcribe_or_gt == '3':
            transcribe_or_gt_string = 'Box coordinate based'
            try:
                try:
                    ground_truth_coords = word_bb_coords(path,file)
                except:
                    pass
                try:
                    ground_truth_coords = iam_word_bb_coords(path,file)
                except:
                    pass
                # ground_truth_coords = alignment_read(file)

                # Transform gt_coords to page_image coords
                for i,box in enumerate(ground_truth_coords):  
                    # print('box: ', box)
                    p1 = np.array([[[box[1], box[3]]]], dtype=np.float32)
                    p2 = np.array([[[box[2], box[4]]]], dtype=np.float32)
                    new_p1 = cv2.perspectiveTransform(p1, M)
                    new_p2 = cv2.perspectiveTransform(p2, M)
                    # print('new_p1: ',new_p1)
                    # print('new_p2: ',new_p2)
                    new_box = [box[0],int(new_p1[0][0][0]),int(new_p2[0][0][0]),int(new_p1[0][0][1]),int(new_p2[0][0][1])]
                    # print('new_box: ', new_box)
                    ground_truth_coords[i] = new_box

            except:
                transcribe_or_gt = None
                print('No bounding box coordinates found.')
                continue
            if ground_truth_coords != None:
                # print('Ground-truth number of words: ',len(ground_truth_coords))

                align_dict = {}
                # -------------DISTANCE SIMILARITY ALIGNMENT / IOU overlap ALIGNMENT------
                # testttt = page_image.copy()
                for word in ground_truth_coords:
                    coord_dist = []
                    overlaps = []
                    for pred_word in word_coords:
                        overlap = get_iou(pred_word, word)
                        overlaps.append(overlap)
                    if sum(overlaps) == 0:
                        continue
                    # print('overlaps: ', overlaps)
                    # print('word: ',word[0])
                    max_overlap_index = overlaps.index(max(overlaps))
                    # if word[0] == '12':
                    #     print('12 ious: ', overlaps)
                    #     cv2.rectangle(testttt, (word_coords[max_overlap_index][1][0],word_coords[max_overlap_index][1][2]),(word_coords[max_overlap_index][1][1],word_coords[max_overlap_index][1][3]),(0,255,0),2)
                    #     cv2.rectangle(testttt, (word[1],word[3]),(word[2],word[4]),(255,0,0),2)
                    #     plt.imshow(testttt,cmap='gray')
                    #     plt.show()
                    # print('gt coords: ', word[1],word[2],word[3],word[4])
                    # print('pred coords: ', word_coords[max_overlap_index])
                    #     x1_diff = abs(pred_word[1][0]-word[1])**2
                    #     x2_diff = abs(pred_word[1][1]-word[2])**2
                    #     y1_diff = abs(pred_word[1][2]-word[3])**2
                    #     y2_diff = abs(pred_word[1][3]-word[4])**2
                    #     p1_dist = np.sqrt(x1_diff+y1_diff)
                    #     p2_dist = np.sqrt(x2_diff+y2_diff)
                    #     p3_dist = np.sqrt(x1_diff+y2_diff)
                    #     p4_dist = np.sqrt(x2_diff+y1_diff)
                    #     coord_dist.append([p1_dist, p2_dist, p3_dist, p4_dist])
                    # min_coord_dist = (min(coord_dist, key=sum))
                    # min_coord_ind = coord_dist.index(min_coord_dist)
                    # print('word_coords[min_coord_ind]:',word_coords[min_coord_ind])
                    if word[0] in align_dict:
                        align_dict[word[0]].append(word_coords[max_overlap_index][0])
                        align_dict[word[0]].append(word_coords[max_overlap_index][1].copy())
                    else:
                        align_dict[word[0]] = word_coords[max_overlap_index].copy()
                break
            else:
                transcribe_or_gt = None
                continue

            # Align based on transcript
        if transcribe_or_gt == '2':
            transcribe_or_gt_string = 'Transcript based'
            try:
                ground_truth_words = get_word_vector(path,file)
            except:
                transcribe_or_gt = None
                print('No transcript .txt file found.')
                continue
            if ground_truth_words != None:
                no_gt_words = 0
                # print('ground_truth_words: ',ground_truth_words)
                for listElem in ground_truth_words:
                    no_gt_words += len(listElem)  
                # print('Ground-truth number of words: ',no_gt_words)
                try:
                    align_dict = {}
                    for i,line in enumerate(line_coords):
                        line_word_boxes = []
                        for word in word_coords:
                            if word[0] == i+1:
                                line_word_boxes.append(word)
                        line_word_boxes.sort(key = lambda x: x[1])
                        # print('line_word_boxes ', i+1, ': ',line_word_boxes)
                        for j,word in enumerate(line_word_boxes):
                                # print('j: ', j)
                                line_number = word[0] - 1
                                # print('line number: ', line_number)
                                key_word = ground_truth_words[line_number][j]
                                # print('line_number:',line_number,' j:',j)
                                # print('ground_truth_words[line_number][j]: ', ground_truth_words[line_number][j])
                                if key_word in align_dict:
                                    align_dict[key_word] += word
                                else:
                                    align_dict[key_word] = word
                    break
                except:
                    transcribe_or_gt = None
                    print('Error: Number of GT and predicted boxes must be equal.')
                    continue
            else:
                transcribe_or_gt = None
                print('No ground truth found.')
                continue

        # MANUAL TRANSCRIPTION AND ALIGNMENT
        if transcribe_or_gt == '1':
            transcribe_or_gt_string = 'Manual Transcription'
            align_dict = {}
            for i,line in enumerate(line_coords):
                line_image = page_image.copy()
                cv2.rectangle(line_image, (int(line[0]),int(line[2])), (int(line[1]),int(line[3])),(0,255,0),4)
                line_word_boxes = []
                for word in word_coords:
                    if word[0] == i+1:
                        cv2.rectangle(line_image, (word[1][0],word[1][2]), (word[1][1], word[1][3]), (0,0,255),4)
                        line_word_boxes.append(word)
                line_word_boxes.sort(key = lambda x: x[1])
                print('Line {}:'.format(i+1),' {} boxes'.format(len(line_word_boxes)), ', ', line_word_boxes)
                resized3 = resize(line_image, scale_percent)
                cv2.imshow('line', resized3)
                cv2.waitKey(1)
                transcript = input('Line {} transcript: '.format(i+1))
                # cv2.destroyAllWindows()
                transcript = transcript.split()
                for j,word in enumerate(line_word_boxes):
                    if len(transcript) > j:
                        if transcript[j] in align_dict:
                            align_dict[transcript[j]] += word
                        else:
                            align_dict[transcript[j]] = word
                    else:
                        pass
            break

    print('ALIGN DICT: ',align_dict)


    try:
        try:
            ground_truth_coords = word_bb_coords(path,file)
        except:
            pass
        try:
            ground_truth_coords = iam_word_bb_coords(path,file)
        except:
            pass
        # ground_truth_coords = alignment_read(file)

        # Transform gt_coords to page_image coords
        for i,box in enumerate(ground_truth_coords):  
            # print('box: ', box)
            p1 = np.array([[[box[1], box[3]]]], dtype=np.float32)
            p2 = np.array([[[box[2], box[4]]]], dtype=np.float32)
            new_p1 = cv2.perspectiveTransform(p1, M)
            new_p2 = cv2.perspectiveTransform(p2, M)
            # print('new_p1: ',new_p1)
            # print('new_p2: ',new_p2)
            new_box = [box[0],int(new_p1[0][0][0]),int(new_p2[0][0][0]),int(new_p1[0][0][1]),int(new_p2[0][0][1])]
            # print('new_box: ', new_box)
            ground_truth_coords[i] = new_box

    except:
        ground_truth_coords = None
        print('Can not calculate IOU metric, no bounding box coordinates found.')
    if ground_truth_coords != None:
        mean_iou = iou_metric(align_dict, ground_truth_coords)
        print('mean IOU = ', mean_iou)


    # Create annotated image
    if remove_word_boxes != []:
        annotated_segmented_image_temp = new_words_image.copy()
    else:
        annotated_segmented_image_temp = words_image.copy()

    annotated_image_temp = page_image.copy()
    pil_image = Image.fromarray(cv2.cvtColor(annotated_image_temp, cv2.COLOR_BGR2RGB))
    pil_image2 = Image.fromarray(cv2.cvtColor(annotated_image_temp, cv2.COLOR_BGR2RGB))
    # pil_image2 = Image.fromarray(cv2.cvtColor(annotated_segmented_image_temp, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 50)
    draw_image = ImageDraw.Draw(pil_image)
    draw_image2 = ImageDraw.Draw(pil_image2)
    visited = []
    for key,val in align_dict.items():
            # print('val: ', val)
            if len(val) == 2:
                if val[1] in visited:
                    # cv2.putText(annotated_image, key, (val[1][0]+250,val[1][2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)
                    draw_image.text((val[1][0],val[1][2]), key, font=font,fill=(255, 0, 0))
                    draw_image2.text((val[1][0],val[1][2]), key, font=font,fill=(255, 0, 0))
                else:
                    # cv2.putText(annotated_image, key, (val[1][0],val[1][2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)
                    draw_image.text((val[1][0],val[1][2]), key, font=font,fill=(255, 0, 0))
                    draw_image2.text((val[1][0],val[1][2]), key, font=font,fill=(255, 0, 0))
                    draw_image2.rectangle([(val[1][0],val[1][2]), (val[1][1],val[1][3])], outline="#00ff00",width=3)
                    visited.append(val[1])
            else:
                for coord in val:
                    if type(coord) == list:
                        if coord in visited:
                            # cv2.putText(annotated_image, key, (coord[0]+250,coord[2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)
                            draw_image.text((coord[0],coord[2]), key, font=font,fill=(255, 0, 0))
                            draw_image2.text((coord[0],coord[2]), key, font=font,fill=(255, 0, 0))
                            # draw_image2.rectangle([(coord[0],coord[2]), (coord[1],coord[3])], outline="#00ff00", width=3)

                        else:
                            # cv2.putText(annotated_image, key, (coord[0],coord[2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)
                            draw_image.text((coord[0],coord[2]), key, font=font,fill=(255, 0, 0))
                            draw_image2.text((coord[0],coord[2]), key, font=font,fill=(255, 0, 0))
                            draw_image2.rectangle([(coord[0],coord[2]), (coord[1],coord[3])], outline="#00ff00", width=3)
                            visited.append(coord)

    annotated_image = np.asarray(pil_image)
    annotated_segmented_image = np.asarray(pil_image2)


    cv2.imwrite('results/{}/annotated_img.png'.format(file), annotated_image)
    cv2.imwrite('results/{}/annotated_segmented_img.png'.format(file), annotated_segmented_image)
    no_of_lines2 = max(align_dict.values())[0]
    align_list = []
    for i in range(1, no_of_lines2+1):
        for key,val in align_dict.items():
            if len(val) > 2:
                no_val = int(len(val))
                for j in range(no_val):
                    if val[j] == i:
                        align_list.append([key,val[j+1]])
            else:
                if val[0] == i:
                    align_list.append([key, val[1]])

    # Calculate how many boxes there are in the final alignment
    align_visited_boxes = []
    for coord in align_dict.values():
        if len(coord) == 2:
            if coord[1] not in align_visited_boxes:
                align_visited_boxes.append(coord[1])
        else:
            for i in coord:
                if type(i) == list:
                    if i not in align_visited_boxes:
                        align_visited_boxes.append(i)

    with open("./results/{}/metrics.txt".format(file), "w", encoding="utf8") as f:
        mean_iou = np.round(mean_iou, 8)
        mean_iou = str(mean_iou).replace('.', ',')
        if transcribe_or_gt == '3':
            try:
                gt_num_words = len(word_bb_coords(path,file))
            except:
                pass
            try:
                gt_num_words = len(iam_word_bb_coords(path,file))
            except:
                pass
            # gt_num_words = len(alignment_read(file))
        else:
            gt_num_words = 'Unknown'
        f.write('file: {} \nMethod: 2 \nt1: {} \nmin_gap: {} \nNo lines: {} \nNo seg boxes: {} \nNo align boxes: {}  \nNo aligned words (alignment): {} \nNo. GT words: {} \nAlignment method: {} \nMean IOU: {}'.format(file, t1, min_gap, no_of_lines, len(word_coords),len(align_visited_boxes),len(align_list),gt_num_words,transcribe_or_gt_string, mean_iou))

    # Draw rectangles on words image from ground truth
    if transcribe_or_gt == '3':
        try:
            ground_truth_coords = word_bb_coords(path,file)
        except:
            pass
        try:
            ground_truth_coords = iam_word_bb_coords(path,file)
        except:
            pass
        # ground_truth_coords = alignment_read(file)
        gt_orig_img = image.copy()
        gt_boxes_img = cv2.cvtColor(page_image.copy(),cv2.COLOR_BGR2RGB)
        gt_words_img = cv2.cvtColor(page_image.copy(),cv2.COLOR_BGR2RGB)
        colors = [(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
        coli = 0
        # Transform gt_coords to page_image coords
        for i,box in enumerate(ground_truth_coords):  
            # print('box: ', box)
            cv2.rectangle(gt_orig_img, (box[1],box[3]), (box[2],box[4]),colors[coli],2)
            p1 = np.array([[[box[1], box[3]]]], dtype=np.float32)
            p2 = np.array([[[box[2], box[4]]]], dtype=np.float32)
            new_p1 = cv2.perspectiveTransform(p1, M)
            new_p2 = cv2.perspectiveTransform(p2, M)
            # print('new_p1: ',new_p1)
            # print('new_p2: ',new_p2)
            new_box = [box[0],int(new_p1[0][0][0]),int(new_p2[0][0][0]),int(new_p1[0][0][1]),int(new_p2[0][0][1])]
            # print('new_box: ', new_box)
            ground_truth_coords[i] = new_box
            cv2.rectangle(gt_words_img, (new_box[1],new_box[3]), (new_box[2],new_box[4]),colors[coli],2)
            cv2.rectangle(gt_boxes_img, (new_box[1],new_box[3]), (new_box[2],new_box[4]),colors[coli],2)
            if coli == 4:
                coli = 0
            else:
                coli += 1

        for x in align_list:
            cv2.rectangle(gt_words_img, (x[1][0],x[1][2]),(x[1][1],x[1][3]),(0,255,0),2)
        cv2.imwrite('results/{}/gt_orig_img.png'.format(file), gt_orig_img)
        cv2.imwrite('results/{}/gt_words_img.png'.format(file), gt_words_img)
        cv2.imwrite('results/{}/gt_boxes_img.png'.format(file), gt_boxes_img)

    shape = page_image.shape
    # save xml file of alignment
    xml_file = xml_output(align_dict,no_of_lines,shape)
    with open("./results/{}/{}.xml".format(file,file), "w",encoding="utf8") as f:
        f.write(xml_file)

    print('no of align boxes: ', len(align_visited_boxes))