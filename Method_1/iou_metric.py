from Method_1.xml_word_coords import word_bb_coords
from Method_1.utils import get_iou

def iou_metric(align_dict, gt_coords):
    no_of_lines = max(align_dict.values())[0]
    align_list = []
    for i in range(1, no_of_lines+1):
        for key,val in align_dict.items():
            if len(val) > 2:
                no_val = int(len(val))
                for j in range(no_val):
                    if val[j] == i:
                        align_list.append([key,val[j+1]])
            else:
                if val[0] == i:
                    align_list.append([key, val[1]])

    IOUS = []
    for word in align_list:
        for gt_word in gt_coords:
            if word[0] == gt_word[0]:
                bb1 = word[1]
                bb2 = gt_word[1:len(gt_word)]
                # print(word[0], ' bb1: ', bb1)
                # print(gt_word[0], ' bb2: ', bb2)
                iou = get_iou(bb1, bb2)
                print(print(word[0],' iou=', iou))
                IOUS.append(iou)
                gt_coords.remove(gt_word)
                break
    print('ious:', IOUS)
    return sum(IOUS)/len(IOUS)