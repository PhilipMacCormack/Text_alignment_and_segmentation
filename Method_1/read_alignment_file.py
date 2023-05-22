import xmltodict
import os

def alignment_read(file):
    with open('results/{}/alto/{}.xml'.format(file,file),'r',encoding='utf8') as f:
        data = f.read()
    dict_data = xmltodict.parse(data)
    word_coords = []
    # print(dict_data)
    for i,tline in enumerate(dict_data['alto']['Layout']['Page']['Textblock']):
        if str(tline)=='String':
            continue
        else:
            for word in dict_data['alto']['Layout']['Page']['Textblock'][tline]:
                if str(word) == 'String':
                    continue
                else:
                    content = dict_data['alto']['Layout']['Page']['Textblock'][tline][word]['@CONTENT']
                    x1 = int(dict_data['alto']['Layout']['Page']['Textblock'][tline][word]['@x1'])
                    x2 = int(dict_data['alto']['Layout']['Page']['Textblock'][tline][word]['@x2'])
                    y1 = int(dict_data['alto']['Layout']['Page']['Textblock'][tline][word]['@y1'])
                    y2 = int(dict_data['alto']['Layout']['Page']['Textblock'][tline][word]['@y2'])
                    word_to_add = [content,x1,x2,y1,y2]
                    word_coords.append(word_to_add)
    return word_coords

# file = 'fac_00178_arsberattelse_1935_sid-06'
# print(alignment_read(file))

# file_path = '../../data/Labours_Memory_Test_Data/'
# files = os.listdir(file_path)

# for file in files:
#     if file[len(file)-4:len(file)] == '.jpg':
#         try:
#             file = file[0:len(file)-4]
#             word_coords = alignment_read(file)
#             # Calculate how many boxes there are in the final alignment
#             align_visited_boxes = []
#             for coord in word_coords:
#                     box = [coord[1],coord[2],coord[3],coord[4]]
#                     if box not in align_visited_boxes:
#                         align_visited_boxes.append(box)
            
#             print('File: ', file)
#             print(len(align_visited_boxes))
#             print('-----------------------------')
#         except:
#             continue