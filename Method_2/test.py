align_dict = {'Årsberättelse': [1, [1178, 1813, 183, 347]], 'Med': [2, [561, 801, 373, 476]], '25': [2, [929, 1093, 400, 484], 8, [473, 680, 1220, 1297]], 'Giresta': [2, [1255, 1545, 380, 483]], 'av': [2, [1635, 1770, 426, 476], 6, [2023, 2174, 989, 1041]], 'andra': [2, [1831, 2125, 400, 482]], 'verksamhetsåret': [2, [2258, 3095, 394, 483]], 'avdelningen': [3, [563, 1224, 500, 703], 4, [2320, 2918, 679, 835]], 'har': [3, [1289, 1482, 542, 624], 6, [1368, 1567, 950, 1038], 9, [3115, 3278, 1366, 1442], 11, [541, 717, 1635, 1717]], 'haft': [3, [1543, 1801, 527, 663]], '12': [3, [2040, 2499, 541, 624]], 'ordinarie': [3, [2040, 2499, 541, 624]], 'möten': [3, [2613, 2978, 540, 618]], 'och': [4, [564, 754, 655, 751], 5, [1476, 1647, 807, 903], 8, [2784, 2962, 1228, 1353]], 'ett': [4, [804, 1020, 632, 757]], 'styrelsesamanträde': [4, [1065, 1984, 679, 850]], 'ur': [4, [2093, 2225, 715, 765]], 'avflyttat': [5, [532, 991, 806, 959]], '3': [5, [1109, 1391, 805, 901], 6, [2268, 2569, 955, 1086]], 'medl': [5, [1109, 1391, 805, 901], 5, [2508, 2790, 826, 899]], 'tillkommit': [5, [1708, 2325, 807, 899]], '11': [5, [2508, 2790, 826, 899]], 'agitasionskomiten': [6, [503, 1293, 944, 1110]], 'beståt': [6, [1618, 1980, 951, 1037]], 'medl,': [6, [2268, 2569, 955, 1086]], 'på': [6, [2642, 2819, 980, 1087]], 'distriktets': [6, [2910, 3383, 942, 1036]], 'årskomferens': [7, [486, 1128, 1070, 1220]], 'som': [7, [1203, 1425, 1121, 1173]], 'hölls': [7, [1501, 1745, 1080, 1176]], 'i': [7, [2025, 2396, 1086, 1230]], 'Upsala': [7, [2025, 2396, 1086, 1230]], 'Folkets': [7, [2503, 2810, 1062, 1178]], 'hus': [7, [2857, 3081, 1083, 1170]], '-': [8, [714, 838, 1228, 1304]], '26': [8, [907, 1124, 1230, 1308]], 'Mars': [8, [907, 1124, 1230, 1308]], '1931.': [8, [1492, 1726, 1212, 1311]], 'Fast': [8, [1492, 1726, 1212, 1311]], 'ombud': [8, [1793, 2171, 1236, 1314]], 'var': [8, [2208, 2389, 1264, 1308]], 'valt': [8, [2452, 2704, 1232, 1311]], 'suplant': [8, [2982, 3409, 1225, 1358]], 'inställde': [9, [474, 920, 1367, 1448]], 'sig': [9, [1025, 1173, 1401, 1523]], 'ingen': [9, [1256, 1555, 1406, 1522]], 'för': [9, [1675, 1841, 1359, 1485]], 'avdelningen.': [9, [1956, 2593, 1371, 1525]], 'År': [9, [2676, 2820, 1366, 1449]], '1931': [9, [2882, 3064, 1369, 1446]], 'krönts': [10, [513, 791, 1495, 1582]], 'med': [10, [883, 1150, 1474, 1584]], 'framgång': [10, [1219, 1680, 1471, 1671]], 'genom': [10, [1747, 2084, 1539, 1667]], 'att': [10, [2162, 2333, 1504, 1582]], 'medlemsantalet': [10, [2360, 3108, 1453, 1592]], 'fördublats.': [11, [819, 1328, 1628, 1752]]}

# align_visited_boxes = []
# for coord in align_dict.values():
#     if len(coord) == 2:
#         if coord[1] not in align_visited_boxes:
#             align_visited_boxes.append(coord[1])
#         else:
#             print('in visited 1', coord)
#     else:
#         for i in coord:
#             if type(i) == list:
#                 if i not in align_visited_boxes:
#                     align_visited_boxes.append(coord[1])
#                 else:
#                     print('in visited 2', i)

# print(len(align_visited_boxes))

visited=[]
for key,val in align_dict.items():
        # print('val: ', val)
        if len(val) == 2:
            if val[1] in visited:
                 pass
                # cv2.putText(annotated_image, key, (val[1][0]+250,val[1][2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)
                # draw_image.text((val[1][0],val[1][2]), key, font=font,fill=(255, 0, 0))
                # draw_image2.text((val[1][0],val[1][2]), key, font=font,fill=(255, 0, 0))
            else:
                # cv2.putText(annotated_image, key, (val[1][0],val[1][2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)
                # draw_image.text((val[1][0],val[1][2]), key, font=font,fill=(255, 0, 0))
                # draw_image2.text((val[1][0],val[1][2]), key, font=font,fill=(255, 0, 0))
                # draw_image2.rectangle([(val[1][0],val[1][2]), (val[1][1],val[1][3])], outline="#00ff00",width=3)
                visited.append(val[1])
        else:
            for coord in val:
                if type(coord) == list:
                    if coord in visited:
                         pass
                        # cv2.putText(annotated_image, key, (coord[0]+250,coord[2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)
                        # draw_image.text((coord[0],coord[2]), key, font=font,fill=(255, 0, 0))
                        # draw_image2.text((coord[0],coord[2]), key, font=font,fill=(255, 0, 0))
                        # draw_image2.rectangle([(coord[0],coord[2]), (coord[1],coord[3])], outline="#00ff00", width=3)

                    else:
                        # cv2.putText(annotated_image, key, (coord[0],coord[2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)
                        # draw_image.text((coord[0],coord[2]), key, font=font,fill=(255, 0, 0))
                        # draw_image2.text((coord[0],coord[2]), key, font=font,fill=(255, 0, 0))
                        # draw_image2.rectangle([(coord[0],coord[2]), (coord[1],coord[3])], outline="#00ff00", width=3)
                        visited.append(coord)
