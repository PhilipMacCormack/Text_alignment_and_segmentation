import numpy as np
import matplotlib.pyplot as plt
import cv2

from Method_1.utils import page_resize, ratio

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
        return cv2.warpPerspective(img, M, (int(width), int(height))), s_points, t_points, M
        
    #s_points: top_left, bot_left, bot_right, top_right
    newImage, s_points, t_points, M = persp_transform(img, page_contour)

    return [newImage,s_points, t_points, M]
