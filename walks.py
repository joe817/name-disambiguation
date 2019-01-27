#generate walks

import sys
import os
import random
from collections import Counter
import numpy as np
import pandas as pd
import pickle

class MetaPathGenerator:
    def __init__(self):
        self.paper_author = dict()
        self.author_paper = dict()
        self.conf_paper = dict()
        self.paper_conf = dict()
        self.paper_word = dict()
        self.word_paper = dict()

    def read_data(self, dirpath):
        temp=set()
        with open(dirpath + "/paper_word.txt") as pafile:
            for line in pafile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_word:
                        self.paper_word[p] = []
                    self.paper_word[p].append(a)
                    if a not in self.word_paper:
                        self.word_paper[a] = []
                    self.word_paper[a].append(p)
        temp.clear()
        


        with open(dirpath + "/paper_conf.txt") as pcfile:
            for line in pcfile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    #if p not in self.paper_conf:
                        #self.paper_conf[p] = []
                    #self.paper_conf[p].append(a)
                    self.paper_conf[p] = a
                    if a not in self.conf_paper:
                        self.conf_paper[a] = []
                    self.conf_paper[a].append(p)
        temp.clear()

        
        
                
        with open(dirpath + "/paper_author.txt") as pafile:
            for line in pafile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_author:
                        self.paper_author[p] = []
                    self.paper_author[p].append(a)
                    if a not in self.author_paper:
                        self.author_paper[a] = []
                    self.author_paper[a].append(p)
        temp.clear()

        print ("#papers ", len(self.paper_conf))
        print ("#confs  ", len(self.conf_paper))       
        print ("#authors", len(self.author_paper))
        print ("#words", len(self.word_paper))
        
    def generate_PHNet(self):
        #print (len(self.paper_word),len(self.paper_author),len(self.paper_conf))
        
        PHNet =  pd.DataFrame(np.zeros((len(self.paper_conf),len(self.paper_conf))),index= list(self.paper_conf.keys()),columns=list(self.paper_conf.keys()))           
        print ("constructing PHNet...")
        
        print ("extracting paper-author...")
        i=0
        for paper in self.paper_author:
            i=i+1
            if i%1000==0:
                print (i,'pairs')
            for author in self.paper_author[paper]:
                for paper1 in self.author_paper[author]:
                    if (paper1!=paper):
                        PHNet[paper][paper1]+=1
                        
        print ("extracting paper-venue...") 
        i=0
        for paper in self.paper_conf:
            i=i+1
            if i%1000==0:
                print (i,'pairs')
            #for conf in self.paper_conf[paper]:
            conf = self.paper_conf[paper]
            if conf!='22':#null
                for paper1 in self.conf_paper[conf]:
                    if (paper1!=paper):
                        PHNet[paper][paper1]+=1
                        
        print ("extracting paper-word...")
        i=0
        for paper in self.paper_word:
            i=i+1
            if i%1000==0:
                print (i,'pairs')
            for word in self.paper_word[paper]:
                for paper1 in self.word_paper[word]:
                    if (paper1!=paper):
                        PHNet[paper][paper1]+=1
 
                        
        with open ("gene/PHNet.pkl",'wb') as file:
            pickle.dump(PHNet,file)
        
        print ("PHNet done")
    

    def generate_WMRW(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        for conf0 in self.conf_paper:
            if conf0!="22":##conf=null
                conf = conf0
                for j in range(0, numwalks): #wnum walks
                    outline = ""
                    for i in range(0, walklength):
                        if conf!="22":
                            papers = self.conf_paper[conf]
                            nump = len(papers)
                            paperid = random.randrange(nump)
                            paper = papers[paperid]
                            outline += " " + paper
                        
                        if paper in self.paper_author:
                            authors = self.paper_author[paper]
                            numa = len(authors)
                            authorid = random.randrange(numa)
                            author = authors[authorid]
                            
                            papers = self.author_paper[author]
                            nump = len(papers)
                            paperid = random.randrange(nump)
                            paper = papers[paperid]
                            outline += " " + paper                

                        
                        if paper in self.paper_word:
                            words = self.paper_word[paper]
                            numw = len(words)
                            wordid = random.randrange(numw) 
                            word = words[wordid]
                        
                            papers = self.word_paper[word]
                            nump = len(papers)
                            paperid = random.randrange(nump)
                            paper = papers[paperid]
                            outline += " " + paper
                        
                        conf = self.paper_conf[paper]
                        
                    outfile.write(outline + "\n")
        outfile.close()
        
        print ("walks done")

numwalks = 2
walklength = 100

dirpath = "gene"
outfilename ="gene/WMRW.txt" #weighted_metapath_random_walk

def main():
    mpg = MetaPathGenerator()
    mpg.read_data(dirpath)
    mpg.generate_PHNet()
    mpg.generate_WMRW(outfilename, numwalks, walklength)


if __name__ == "__main__":
    main()