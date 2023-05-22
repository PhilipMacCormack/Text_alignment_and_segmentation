import xmltodict
import cv2
import matplotlib.pyplot as plt
from Method_2.page_segment import page_segment

## SCRIPT FOR READING XML FILES FOR ONE PAGE IN LABOURS MEMORY, RETURNS A LIST OF BOUNDING BOX COORDS FOR ALL THE WORDS IN THE PAGE

def word_bb_coords(path,file):
    with open('{}gt/{}.xml'.format(path,file),'r',encoding='utf8') as f:
        data = f.read()
    dict_data = xmltodict.parse(data)
    word_coords = []
    for i,tline in enumerate(dict_data['alto']['Layout']['Page']['PrintSpace']['TextBlock']['TextLine']):
        try:
            if i==0:
                content1 = tline['String']['@CONTENT']
                word_coords.append([content1, int(tline['String']['@HPOS']),int(tline['String']['@HPOS'])+int(tline['String']['@WIDTH']), int(tline['String']['@VPOS']), int(tline['String']['@VPOS'])+int(tline['String']['@HEIGHT'])])
        except:
            pass
        for string in tline['String']:
            if type(string) is dict:
                height = int(string['@HEIGHT'])
                width = int(string['@WIDTH'])
                vpos = int(string['@VPOS'])
                hpos = int(string['@HPOS'])
                content = string['@CONTENT']
                word_coords.append([content, hpos,hpos+width,vpos,vpos+height])

    return word_coords

def iam_word_bb_coords(path,file):
    with open('{}gt/{}.xml'.format(path,file),'r',encoding='utf8') as f:
        data = f.read()
    dict_data = xmltodict.parse(data)
    word_coords = []
    for i,tline in enumerate(dict_data['form']['handwritten-part']['line']):
        # print(tline)
        for string in tline['word']:
            # print(string)
            if type(string['cmp']) is dict:
                # print(string)
                height = int(string['cmp']['@height'])
                width = int(string['cmp']['@width'])
                y = int(string['cmp']['@y'])
                x = int(string['cmp']['@x'])
                content = string['@text']
                # print(content)
                # print(string['cmp'])

                word_coords.append([content, x, x+width, y, y+height])
            if type(string['cmp']) is list:
                content = string['@text']
                # print(content)
                x1_list = []
                x2_list = []
                y1_list = []
                y2_list = []
                for dictt in string['cmp']:
                    x1_list.append(int(dictt['@x']))
                    x2_list.append(int(dictt['@x'])+int(dictt['@width']))
                    y1_list.append(int(dictt['@y']))
                    y2_list.append(int(dictt['@y'])+int(dictt['@height']))
                    # print(dictt)
                x1 = min(x1_list)
                x2 = max(x2_list)
                y1 = min(y1_list)
                y2 = max(y2_list)
                word_coords.append([content, x1,x2,y1,y2])
                
    return word_coords
