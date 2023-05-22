from bs4 import BeautifulSoup
import cv2
import matplotlib.pyplot as plt

## SCRIPT FOR READING XML FILES FOR ONE PAGE IN LABOURS MEMORY, SHOWS BOUNDING BOX FOR TEXT REGION AND BOUNDING BOXES FOR TEXT LINES

with open('./data/lab4.xml','r') as f:
    data = f.read()

bs_data = BeautifulSoup(data, "xml")
allText = bs_data.findAll('Unicode')

for i in range(len(allText)):
    if i == len(allText)-1:
        text = allText[i]

final_text = []
known = set([])
corpus = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖabcdefghijklmnopqrstuvwxyzåäö1234567890!#%&/?'

for tex in text:
    # print(tex)
    word = ''
    for index, token in enumerate(tex):
        print('|',token,'|')
        if token in known:
            continue
        if token == '.' or token == ' ':
            final_text.append(word)
            word = ''
        if token == 'Ã' and tex[index+1] == '…':   # Å
            tokk = 'Å'
            word += tokk
            known.add(tex[index+1])
        # if token == '.' and tex[index+1] == '.':   # Ä
        #     tokk = 'Ä'
        #     word += tokk
        #     known.add(tex[index+1])
        # if token == 'Ã' and tex[index+1] == '…':   # Ö
        #     tokk = 'Ö'
        #     word += tokk
        #     known.add(tex[index+1])
        if token == 'Ã' and tex[index+1] == '¥':   # å
            tokk = 'å'
            word += tokk
            known.add(tex[index+1])
        if token == 'Ã' and tex[index+1] == '¤':   # ä
            tokk = 'ä'
            word += tokk
            known.add(tex[index+1])
        if token == 'Ã' and tex[index+1] == '¶':   # ö
            tokk = 'ö'
            word += tokk
            known.add(tex[index+1])
        # if token == '/' and tex[index+1] == 'r' and tex[index+2] == '/' and tex[index+3] == 'n':
        #     continue
        # if token == 'r' and tex[index-1] == '/' and tex[index+1] == '/'  and tex[index+2] == 'n':
        #     continue
        # if token == '/' and tex[index-1] == 'r' and tex[index-2] == '/'  and tex[index+1] == 'n':
        #     continue
        if token == 'n' and tex[index-1] == '/' and tex[index-2] == 'r'  and tex[index-3] == '/' and tex[index+1] != '-':
            word = ''
            continue
        if token == 'n' and tex[index-1] == '/' and tex[index-2] == 'r'  and tex[index-3] == '/' and tex[index+1] == '-':
            continue
        elif token in corpus:
            tokk = token
            word += tokk
print(final_text)