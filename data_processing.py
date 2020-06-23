#data processing
import pickle
from gensim.models import word2vec
import networkx as nx 
from sklearn.cluster import AgglomerativeClustering
import  xml.dom.minidom
import xml.etree.ElementTree as ET
import re
import os
import numpy as np 


path = 'raw-data/'
file_names = os.listdir(path)

r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～]+'
stopword = ['at','based','in','of','for','on','and','to','an','using','with','the']

keyid=0
papers = {}
authors = {}
jconfs = {}
word={}

for fname in file_names:
    f = open(path + fname,'r',encoding = 'utf-8').read()
    text=re.sub(u"&",u" ",f)
    root = ET.fromstring(text)

    for i in root.findall('publication'):
        paper = i.find('title').text
        pid = i.find('id').text
        papers[pid] = paper
        
    
for fname in file_names:
    f = open(path + fname,'r',encoding = 'utf-8').read()
    text=re.sub(u"&",u" ",f)
    root = ET.fromstring(text)

    for i in root.findall('publication'):
        jconf = i.find('jconf').text.strip().replace(" ", "")
        if jconf not in jconfs:
            jconfs[jconf] = keyid
            keyid = keyid + 1

authorid=0
author1={}            
for fname in file_names:
    f = open(path + fname,'r',encoding = 'utf-8').read()
    text=re.sub(u"&",u" ",f)
    root = ET.fromstring(text)

    for i in root.findall('publication'):
        authorlist = i.find('authors').text.strip().split(",")
        for author in authorlist:
            author = author.replace(" ", "")
            if author not in authors:
                authors[author] = keyid
                keyid = keyid + 1 
    if fname not in author1:
        author1[fname] = authorid
        authorid = authorid + 1

        
for fname in file_names:
    f = open(path + fname,'r',encoding = 'utf-8').read()
    text=re.sub(u"&",u" ",f)
    root = ET.fromstring(text)
    
    for i in root.findall('publication'):
        pid = i.find('id').text
        line = i.find('title').text
        line = re.sub(r, ' ', line)
        line = line.replace('\t',' ')
        line = line.lower()
            #f1.write(line+'\n')

        split_cut = line.split(' ')
        for j in split_cut:
            if len(j)>1 and (j not in stopword):
                if j not in word:
                    word[j] = 1
                else:
                    word[j] = word[j] +1

                    
                    
f1 = open ('gene/paper_author.txt','w',encoding = 'utf-8')
f2 = open ('gene/paper_conf.txt','w',encoding = 'utf-8')
f3 = open ('gene/paper_word.txt','w',encoding = 'utf-8')

f4 = open ('gene/paper_author1.txt','w',encoding = 'utf-8')
f5 = open ('gene/paper_title.txt','w',encoding = 'utf-8')


for fname in file_names:
    f = open(path + fname,'r',encoding = 'utf-8').read()
    text=re.sub(u"&",u" ",f)
    root = ET.fromstring(text)

    for i in root.findall('publication'):
        pid = i.find('id').text
        authorlist = i.find('authors').text.strip().split(",")
        jconf = i.find('jconf').text.strip().replace(" ", "")
        f4.write('i'+pid + '\t' + str(author1[fname]) + '\n')
        for author in authorlist:
            if author!=fname[:-4]:
                if (author+'.xml') in author1:
                    f4.write('i'+pid + '\t' + str(author1[author+'.xml']) + '\n')
                author = author.replace(" ", "")
                f1.write('i'+pid + '\t' + str(authors[author]) + '\n')

        f2.write('i'+pid + '\t' + str(jconfs[jconf]) + '\n')
        
        line = i.find('title').text
        line = re.sub(r, ' ', line)
        line = line.replace('\t',' ')
        line = line.lower()
        f5.write('i' + pid +'\t' + line + '\n')
            
        split_cut = line.split(' ')
        for j in split_cut:
            if (j in word)and (word[j]>=2):
                f3.write('i' + pid +'\t' + j + '\n')
                



f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

print(len(author1),"ambiguous names.")