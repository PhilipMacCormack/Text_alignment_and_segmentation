from xml.dom import minidom
from datetime import datetime

def xml_output(align_dict,no_of_lines,shape):
    '''
    align_dict : dict = {word  : [page_number, [x1,x2,y1,y2]]} # The dictionary with words, line number and word bounding box coordinates

    no_of_lines : int = x                                      # An integer number representing the number of lines in the document

    shape : tuple = (height, width)                            # The shape of the document in pixels
    '''
    
    time = datetime.today()
    dt_string = time.strftime("%Y-%m-%d %H:%M:%S")
    root = minidom.Document()
    alto = root.createElement('alto')
    root.appendChild(alto)
    description = root.createElement('Description')
    alto.appendChild(description)
    measurementunit = root.createElement('MeasurementUnit')
    description.appendChild(measurementunit)
    pixel = root.createTextNode('pixel')
    measurementunit.appendChild(pixel)
    processingdatetime = root.createElement('processingDateTime')
    description.appendChild(processingdatetime)
    timenode = root.createTextNode(dt_string)
    processingdatetime.appendChild(timenode)
    layout = root.createElement('Layout')
    alto.appendChild(layout)
    page = root.createElement('Page')
    page.setAttribute('HEIGHT', str(shape[0]))
    page.setAttribute('WIDTH', str(shape[1]))
    layout.appendChild(page)
    block = root.createElement('Textblock')
    page.appendChild(block)
    string = root.createElement('String')
    text = ''
    for i,key in enumerate(align_dict.keys()):
        if i == len(align_dict.keys())-1:
            text += key
        else:
            text += key + ' '
    string.setAttribute('CONTENT', text)
    block.appendChild(string)

    align_list = []
    for i in range(1, no_of_lines+1):
        line_list = []
        for key,val in align_dict.items():
            if len(val) > 2:
                no_val = int(len(val))
                for j in range(no_val):
                    if val[j] == i:
                        line_list.append([key,val[j+1]])
            else:
                if val[0] == i:
                    line_list.append([key, val[1]])
        line_list.sort(key = lambda x: x[1])
        align_list.append(line_list)
    print('align_list: ', align_list)
    for i,line in enumerate(align_list):
        line_no = i+1
        line_i = root.createElement('line{}'.format(line_no))
        block.appendChild(line_i)
        line_text = ''
        string2 = root.createElement('String')
        for j,word in enumerate(line):
            if j == len(line)-1:
                line_text += word[0]
            else:
                line_text += word[0] + ' '
            word_j = root.createElement('word{}'.format(j+1))
            word_j.setAttribute('CONTENT',str(word[0]))
            word_j.setAttribute('x1',str(word[1][0]))
            word_j.setAttribute('x2',str(word[1][1]))
            word_j.setAttribute('y1',str(word[1][2]))
            word_j.setAttribute('y2',str(word[1][3]))
            line_i.appendChild(word_j)
            line_i.appendChild
        string2.setAttribute('CONTENT', line_text)
        line_i.appendChild(string2)

    return root.toprettyxml()


# shape = (5623, 3551)
# dictt = {'Årsberättelse': [1, [1178, 1813, 183, 347]], 'Med': [2, [561, 801, 373, 476]], '25': [2, [929, 1093, 400, 484], 8, [473, 680, 1220, 1297]], 'Giresta': [2, [1255, 1545, 380, 483]], 'av': [2, [1635, 1770, 426, 476], 6, [2268, 2569, 955, 1086]], 'andra': [2, [1831, 2125, 400, 482]], 'verksamhetsåret': [2, [2258, 3095, 394, 483]], 'avdelningen': [3, [563, 1224, 500, 703], 4, [2320, 2918, 679, 835]], 'har': [3, [1543, 1801, 527, 663], 6, [1618, 1980, 951, 1037], 9, [3115, 3278, 1366, 1442], 11, [541, 717, 1635, 1717]], 'haft': [3, [1543, 1801, 527, 663], 6, [1618, 1980, 951, 1037], 9, [3115, 3278, 1366, 1442], 11, [541, 717, 1635, 1717]], '12': [3, [2040, 2499, 541, 624]], 'ordinarie': [3, [2040, 2499, 541, 624]], 'möten': [3, [2613, 2978, 540, 618]], 'och': [4, [564, 754, 655, 751], 5, [1476, 1647, 807, 903], 8, [2784, 2962, 1228, 1353]], 'ett': [4, [804, 1020, 632, 757]], 'styrelsesamanträde': [4, [1065, 1984, 679, 850]], 'ur': [4, [2320, 2918, 679, 835]], 'avflyttat': [5, [532, 991, 806, 959]], '3': [5, [1109, 1391, 805, 901], 6, [2268, 2569, 955, 1086]], 'medl': [5, [1476, 1647, 807, 903], 5, [2508, 2790, 826, 899]], 'tillkommit': [5, [1708, 2325, 807, 899]], '11': [5, [2508, 2790, 826, 899]], 'agitasionskomiten': [6, [503, 1293, 944, 1110]], 'beståt': [6, [1618, 1980, 951, 1037]], 'medl,': [6, [2268, 2569, 955, 1086]], 'på': [6, [2642, 2819, 980, 1087]], 'distriktets': [6, [2910, 3383, 942, 1036]], 'årskomferens': [7, [486, 1128, 1070, 1220]], 'som': [7, [1501, 1745, 1080, 1176]], 'hölls': [7, [1501, 1745, 1080, 1176]], 'i': [7, [2025, 2396, 1086, 1230]], 'Upsala': [7, [2025, 2396, 1086, 1230]], 'Folkets': [7, [2503, 2810, 1062, 1178]], 'hus': [7, [2857, 3081, 1083, 1170]], '-': [8, [714, 838, 1228, 1304]], '26': [8, [907, 1124, 1230, 1308]], 'Mars': [8, [1168, 1347, 1230, 1309]], '1931.': [8, [1492, 1726, 1212, 1311]], 'Fast': [8, [1793, 2171, 1236, 1314]], 'ombud': [8, [1793, 2171, 1236, 1314]], 'var': [8, [2452, 2704, 1232, 1311]], 'valt': [8, [2452, 2704, 1232, 1311]], 'suplant': [8, [2982, 3409, 1225, 1358]], 'inställde': [9, [474, 920, 1367, 1448]], 'sig': [9, [1256, 1555, 1406, 1522]], 'ingen': [9, [1256, 1555, 1406, 1522]], 'för': [9, [1675, 1841, 1359, 1485]], 'avdelningen.': [9, [1956, 2593, 1371, 1525]], 'År': [9, [2676, 2820, 1366, 1449]], '1931': [9, [2882, 3064, 1369, 1446]], 'krönts': [10, [513, 791, 1495, 1582]], 'med': [10, [883, 1150, 1474, 1584]], 'framgång': [10, [1219, 1680, 1471, 1671]], 'genom': [10, [1747, 2084, 1539, 1667]], 'att': [10, [2162, 2333, 1504, 1582]], 'medlemsantalet': [10, [2360, 3108, 1453, 1592]], 'fördublats.': [11, [819, 1328, 1628, 1752]]}
# xml = xml_output(dictt,11,shape)

# with open("testfile.xml", "w",encoding="utf8") as f:
#     f.write(xml)