#gru based deep representation learning

import collections
import math
import os
import sys
import random
from gensim.models import word2vec
import pickle
from sklearn.cluster import AgglomerativeClustering
import  xml.dom.minidom
import xml.etree.ElementTree as ET
import re
import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
import time
from keras.utils import to_categorical



num_step = 20 #GRU时序个数
word_input = 100

paperid_title = {}
with open("gene/paper_title.txt",encoding = 'utf-8') as adictfile:
    for line in adictfile:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paperid_title[toks[0]] = toks[1]

save_model_name = "gene/word2vec.model"
model = word2vec.Word2Vec.load(save_model_name)

paper_vec={}
paper_len={}
for paperid in paperid_title:
    split_cut = paperid_title[paperid].split()
    words_vec = []
    for j in split_cut:
        if (len(words_vec)<num_step) and (j in model):
            words_vec.append(model[j])
    if len(words_vec)==0:
        words_vec.append(2*np.random.random(word_input)-1)
    lenth = len(words_vec)
    for ii in range(len(words_vec),num_step):
        words_vec.append(np.zeros(word_input))
    words_vec = np.array(words_vec)
    paper_vec[paperid] = words_vec
    paper_len[paperid] = lenth
#print (len(paper_vec))    

paper_author = {}
with open("gene/paper_author1.txt",encoding = 'utf-8') as adictfile:
    for line in adictfile:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paper_author[toks[0]] = int(toks[1])

            
paper_jconf = {}
with open("gene/paper_conf.txt",encoding = 'utf-8') as adictfile:
    for line in adictfile:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paper_jconf[toks[0]] = int(toks[1])

            
            
paper = []
paper_lens = []
paper_label = []
paper_labelj = []

for key in paper_vec:
    paper.append(paper_vec[key])
    paper_lens.append(paper_len[key])
    paper_label.append(paper_author[key])
    paper_labelj.append(paper_jconf[key])

paper_label = to_categorical(paper_label)
paper_labelj = to_categorical(paper_labelj)

n_class = len(paper_label[0])
n_classj = len(paper_labelj[0])

data_size = len(paper_vec)
print ('data_size',data_size)
p = np.random.permutation(data_size)
paper1 = []
paper_lens1 = []
paper_label1 = []
paper_labelj1 = []
for i in p:
    paper1.append(paper[i])
    paper_lens1.append(paper_lens[i])
    paper_label1.append(paper_label[i])
    paper_labelj1.append(paper_labelj[i])


def batch_gen(X,X_len,Y,Y1,batch_size):  # 定义batch数据生成器
    idx = 0
    while True:
        if idx+batch_size>len(X):
            idx=0
        start = idx
        idx += batch_size
        yield X[start:start+batch_size],X_len[start:start+batch_size],Y[start:start+batch_size],Y1[start:start+batch_size]


batch_size = 128
embedding_size = 64    # Dimension of the embedding vector.
learning_rate = 0.001
gen = batch_gen(paper1,paper_lens1,paper_label1,paper_labelj1,batch_size)


graph = tf.Graph()

with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.float32, shape=[None,num_step,word_input])
        seq_lengths =  tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, n_class])
        train_labelsj = tf.placeholder(tf.int32, shape=[None, n_classj])



    with tf.device('/cpu:0'):
        with tf.name_scope('embeddings'):
            with tf.variable_scope("gru") as scope:
                gru = tf.contrib.rnn.GRUCell(embedding_size)
                outputs,last  = tf.nn.dynamic_rnn(gru,train_inputs,seq_lengths,dtype=tf.float32)
                embed = tf.reduce_sum(outputs,1)/tf.expand_dims(tf.cast(seq_lengths,tf.float32),1)
                pred = tf.contrib.layers.fully_connected(embed,n_class,activation_fn = None)
                predj = tf.contrib.layers.fully_connected(embed,n_classj,activation_fn = None)


    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=train_labels))+tf.reduce_mean(
                                              tf.nn.softmax_cross_entropy_with_logits(logits=predj, labels=train_labelsj))


    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(train_labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        correct_predj = tf.equal(tf.argmax(predj,1), tf.argmax(train_labelsj,1))
        accuracyj = tf.reduce_mean(tf.cast(correct_predj, tf.float32))

    # Add variable initializer.
    init = tf.global_variables_initializer()



num_steps = 3001
max_F1 = 0
with tf.Session(graph=graph) as session:

    init.run()
    print('Initialized')
    start_time = time.time()
    average_loss = 0
    average_acc = 0
    for step in xrange(num_steps):
        batch_inputs, batch_len,batch_labels,batch_labelsj= next(gen)
        feed_dict = {train_inputs: batch_inputs, seq_lengths:batch_len, train_labels: batch_labels,train_labelsj: batch_labelsj}



        _,loss_val,acc_val = session.run(
                [optimizer,loss,accuracy],
                feed_dict=feed_dict)
        average_loss += loss_val
        average_acc += acc_val


        if step % 100 == 0:
            step_time = time.time()
            if step > 0:
                average_loss /= 100
                average_acc /= 100

            
            if step==0:
                print ('step:',step,' loss:%.3f'%average_loss,' acc:%.3f'%average_acc)
            else:
                print ('step:',step,' loss:%.3f'%average_loss,' acc:%.3f'%average_acc,' cost_time:%.3f'%((step_time-start_time)/60),'m, stop_in:%.3f'%((step_time-start_time)/60*(num_steps-step)/step),'m')
            average_loss = 0
            average_acc = 0



    paper_finalvec = {}
    for i in paper_vec:
        feed_dict ={train_inputs: [paper_vec[i]],seq_lengths: [paper_len[i]]}
        i_output = session.run(outputs,feed_dict = feed_dict)
        embedded = (i_output[0].sum(axis=0)/paper_len[i])
        paper_finalvec[i] = embedded


with open('gene/gruemb.pkl', "wb") as file_obj:  
    pickle.dump(paper_finalvec, file_obj)
