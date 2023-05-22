from bs4 import BeautifulSoup
import cv2
import matplotlib.pyplot as plt

## SCRIPT FOR READING XML FILES FOR ONE PAGE IN LABOURS MEMORY, SHOWS BOUNDING BOX FOR TEXT REGION AND BOUNDING BOXES FOR TEXT LINES

with open('./data/lab4.xml','r') as f:
    data = f.read()
img = cv2.imread('data/lab4.jpg')

bs_data = BeautifulSoup(data,"xml")
region_lines = bs_data.find_all('Coords')
baseline = bs_data.find_all('Baseline')

def Point_show(img,points):
    final_img = img.copy()
    for i in points:
        # print('Points: ',i.get('points'))
        coordlist = []
        medistr = ''
        for ind,j in enumerate(i.get('points')):
            # line_img = img.copy()
            last = len(i.get('points'))-1
            if ind == last:
                medistr += j
                mediint = int(medistr)
                coordlist.append(mediint)
                medistr = ''
            elif j.isnumeric():
                medistr += j
            else:
                mediint = int(medistr)
                coordlist.append(mediint)
                medistr = ''
        for i in range(0,len(coordlist),2):
            if i+3 > len(coordlist)-1:
                break
            p1 = (coordlist[i],coordlist[i+1])
            p2 = (coordlist[i+2],coordlist[i+3])
            if i == 0:
                startcoord = p1
            if i == len(coordlist)-4:
                endcoord = p2
                cv2.line(final_img, startcoord, endcoord, (0,255,0),3)
                # cv2.line(line_img, startcoord, endcoord, (0,255,0),3)
            cv2.line(final_img, p2, p1, (0,255,0),3)
            # cv2.line(line_img, p2, p1, (0,255,0),3)
        # plt.imshow(line_img,cmap='gray')
        # plt.show()
        
    plt.imshow(final_img,cmap='gray')
    plt.show()


Point_show(img,region_lines)
            