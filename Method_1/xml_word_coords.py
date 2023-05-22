import xmltodict

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

# print(word_bb_coords('../../Data/Labours_Memory_Test_Data/','fac_03008_verksamhetsberattelse_1960_sid-02_1'))

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
                print(content)
                print(string['cmp'])

                word_coords.append([content, x, x+width, y, y+height])
            if type(string['cmp']) is list:
                content = string['@text']
                print(content)
                x1_list = []
                x2_list = []
                y1_list = []
                y2_list = []
                for dictt in string['cmp']:
                    x1_list.append(int(dictt['@x']))
                    x2_list.append(int(dictt['@x'])+int(dictt['@width']))
                    y1_list.append(int(dictt['@y']))
                    y2_list.append(int(dictt['@y'])+int(dictt['@height']))
                    print(dictt)
                x1 = min(x1_list)
                x2 = max(x2_list)
                y1 = min(y1_list)
                y2 = max(y2_list)
                word_coords.append([content, x1,x2,y1,y2])
                
    return word_coords

# path = '../../../../Data/iam/'
# file = 'a01-000u'

# img = cv2.imread('{}forms/{}.png'.format(path,file))
# word_coords = iam_word_bb_coords(path,file,img)
# for coord in word_coords:
#     cv2.rectangle(img, (coord[1],coord[3]),(coord[2],coord[4]),(0,255,0),2)

# plt.imshow(img,cmap='gray')
# plt.show()

# img = cv2.imread('./data/fac.jpg')
# img2 = page_segment(img)[0]
# print(page_segment(img)[1])
# print(page_segment(img)[2])
# colors = [(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
# coli=0
# for box in ground_truth_coords:
#     cv2.rectangle(img, (box[0],box[2]), (box[1],box[3]),colors[coli],2)
#     if coli == 4:
#         coli = 0
#     else:
#         coli += 1

# i=1
# for box in ground_truth_coords:
#     cv2.rectangle(img2, (box[0]-int(page_segment(img)[1][0][0]),box[2]-(int(page_segment(img)[1][1][1])-int(page_segment(img)[1][2][1]))), (box[1]-int(page_segment(img)[1][0][0]),box[3]-8),(0,0,255),2)
#     # print(box)
#     # if i == 2:
#     #     break
#     # i+=1

# fig = plt.figure(figsize=(10,10))
# fig.add_subplot(2,1,1)
# plt.title('Ground truth')
# plt.imshow(img[0:1000,0:3000], cmap='gray')
# fig.add_subplot(2,1,2)
# plt.title('Test')
# plt.imshow(img2[0:1000,0:3000],cmap='gray')
# plt.show()

# plt.imshow(img,cmap='gray')
# plt.show()