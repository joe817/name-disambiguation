#weighted heterogeneous network embedding

import collections
import math
import os
import sys
import random
from gensim.models import word2vec
import time
import pickle
from sklearn.cluster import AgglomerativeClustering
import  xml.dom.minidom
import xml.etree.ElementTree as ET
import re
import numpy as np
from six.moves import xrange
import tensorflow as tf




def read_data(filename):
    with open (filename,'r',encoding = 'utf-8') as f:
        data = tf.compat.as_str(f.read()).split()
    return data
vocabulary = read_data('gene/WMRW.txt')
print('Data size', len(vocabulary))


vocabulary_size = len(set(vocabulary))
print ('publication size',vocabulary_size)

def build_dataset(words, n_words):
    count = []
    count.extend(collections.Counter(words).most_common(n_words))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(
        vocabulary, vocabulary_size)
del vocabulary    # Hint to reduce memory.

data_index = 0
vocabulary_len = len(data)


with open('gene/gruemb.pkl', "rb") as file_obj:  
    gruemb = pickle.load(file_obj) 
Wv2e = []
for i in range(vocabulary_size):
    Wv2e.append(gruemb[reverse_dictionary[i]])


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    neg_labels = np.ndarray(shape=(batch_size), dtype=np.int32)
    span = 2 * skip_window + 1    # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)    # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
            k = random.randint(0,vocabulary_len-1)#
            while data[k] in buffer:
                k = random.randint(0,vocabulary_len-1)
            neg_labels[i * num_skips + j] = data[k]#
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels,neg_labels


batch_size = 128
embedding_size = 64    # Dimension of the embedding vector.
skip_window = 1    # How many words to consider left and right.
num_skips = 2   # How many times to reuse an input to generate a label.
num_sampled = 5    # Number of negative examples to sample.



graph = tf.Graph()

with graph.as_default():


    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        neg_sample = tf.placeholder(tf.int32, shape=[batch_size])


    with tf.device('/cpu:0'):
        with tf.name_scope('embeddings'):

            embeddings = tf.Variable(np.array(Wv2e))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            out_embed = tf.nn.embedding_lookup(embeddings, tf.reduce_sum(train_labels, axis=1))
            cosin = tf.divide(tf.reduce_sum(tf.multiply(embed, out_embed), axis=1), tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(embed), axis=1)), 
                                                                              tf.sqrt(tf.reduce_sum(tf.square(out_embed), axis=1))))

            neg_embed = tf.nn.embedding_lookup(embeddings, neg_sample)

            cosin1 = tf.divide(tf.reduce_sum(tf.multiply(embed, neg_embed), axis=1), tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(embed), axis=1)), 
                                                                              tf.sqrt(tf.reduce_sum(tf.square(neg_embed), axis=1))))            

        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                    tf.truncated_normal(
                            [vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=num_sampled,
                        num_classes=vocabulary_size))-tf.reduce_mean(cosin-cosin1)#-tf.reduce_mean(cosin-cosin1)
 

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

#Begin training.
num_steps = 60001 
max_F1 = 0
with tf.Session(graph=graph) as session:

    init.run()
    print('Initialized')
    start_time = time.time()
    average_loss = 0
    
    for step in xrange(num_steps):
        batch_inputs, batch_labels,neg_labels= generate_batch(batch_size, num_skips,
                                                                                                skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels,neg_sample:neg_labels}


        _, loss_val = session.run(
                [optimizer, loss],
                feed_dict=feed_dict)
        average_loss += loss_val


        if step % 2000 == 0:
            step_time = time.time()
            if step > 0:
                average_loss /= 2000

            if step==0:
                print ('Average loss at step ',step,': ',average_loss)
            else:
                print ('Average loss at step ',step,': ',average_loss,' cost_time:%.3f'%((step_time-start_time)/60),'m, stop_in:%.3f'%((step_time-start_time)/60*(num_steps-step)/step),'m')
            average_loss = 0

    final_embeddings = normalized_embeddings.eval()




pembd = {} 
for i in range(len(final_embeddings)):
    pembd[reverse_dictionary[i]] = final_embeddings[i]

with open('final_emb/pemb_final.pkl', "wb") as file_obj:  
    pickle.dump(pembd, file_obj)    
