import numpy as np
import cv2


def page_border(img):
    """
    Crops an image of a page to the borders of said page.
    Credit: https://stackoverflow.com/questions/60145395/crop-exact-document-paper-from-image-by-removing-black-border-from-photos-in-jav
    Parameters:
        img (np.ndarray): The image to border
    """
    # Blur the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        gaussian_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours and sort for largest contour
    cnts = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    error = True
    use_cnt = None
    display_cnt = None
    no_border = False

    # Sorted by area, find the largest contour which is approximately rectangular
    # Crop to this largest approximate rectangle
    for c in cnts:
        # Perform contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Check if a rectangle
        if len(approx) == 4:
            error = False
            display_cnt = approx.reshape(4, 2)

            x1 = display_cnt[0][0] if (display_cnt[0][0] > display_cnt[1][0]) else display_cnt[1][0]
            y1 = display_cnt[0][1] if (display_cnt[0][1] > display_cnt[3][1]) else display_cnt[3][1]

            x2 = display_cnt[2][0] if (display_cnt[2][0] < display_cnt[3][0]) else display_cnt[3][0]
            y2 = display_cnt[1][1] if (display_cnt[1][1] < display_cnt[2][1]) else display_cnt[2][1]
            img = img[y1: y2, x1: x2]
            break

    return error, img