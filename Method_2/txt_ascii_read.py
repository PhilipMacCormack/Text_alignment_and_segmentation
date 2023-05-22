
import linecache

def get_word_vector(path,file):
    text_vector = []
    with open('{}txt/{}.txt'.format(path,file),'r',encoding='utf8') as f:
        # text = f.read()
        no_lines = len(f.readlines())
    for i in range(no_lines):
        line = linecache.getline(r"{}txt/{}.txt".format(path,file),i+1).strip().split()
        text_vector.append(line)
        # utf8_vector = text.split()
    # return utf8_vector
    return text_vector

def get_line_vector(path,file,line):
    with open('{}txt/{}.txt'.format(path,file),'r',encoding='utf8') as f:
        text = f.read()
        lines = text.splitlines()
        for i,words in enumerate(lines):
            if i+1 == line:
                res_line = words.split()
    return res_line

# path = '../../../Data/Labours_Memory/export_job_2576375/776034/Örsundsbro,_Giresta_avd_025/'
# file = 'fac_03008_arsberattelse_1931'

# vec = get_word_vector(path, file)
# print(vec)
# count = 0
# for listElem in vec:
#     count += len(listElem)  

# print(count)